/*
 * pulp_nn_conv_u8_u8_i8.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"
#include "pulp_nn_utils.h"
#include "pulp_nn_kernels.h"


void pulp_nn_conv_u8_u8_i8(
                        uint8_t *pIn,                 // pointer to INPUT TENSOR
                        uint8_t *pIm2ColBuffer,       // pointer to pre-allocated L1 buffer hosting IM2COL BUFFER
                        int8_t *pBias,                // pointer to BIAS
                        uint8_t *pOut,                // pointer to OUTPUT TENSOR
                        int8_t *pWeight,              // pointer to WEIGHT TENSOR
                        int32_t *pKappa,              // pointer to PER-CHANNEL SCALING FACTOR
                        int32_t *pLambda,             // pointer to PER-CHANNEL OFFSET FACTOR
                        uint16_t out_mult,            // SCALAR MULTIPLIER
                        uint16_t out_shift,           // SCALAR RIGHT-SHIFT
                        uint16_t dim_in_x,            // input tensor WIDTH
                        uint16_t dim_in_y,            // input tensor HEIGHT
                        uint16_t ch_in,               // input tensor CHANNEL_IN
                        uint16_t dim_out_x,           // output tensor WIDTH
                        uint16_t dim_out_y,           // output tensor HEIGHT
                        uint16_t ch_out,              // output tensor CHANNEL_OUT
                        uint16_t dim_kernel_x,        // weight tensor FILTER WIDTH
                        uint16_t dim_kernel_y,        // weight tensor FILTER HEIGHT
                        uint16_t padding_y_top,
                        uint16_t padding_y_bottom,
                        uint16_t padding_x_left,
                        uint16_t padding_x_right,
                        uint16_t stride_x,
                        uint16_t stride_y,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
  uint16_t ch_in_r = ch_in;
  uint16_t ch_out_r = ch_out;

  int core_id = pi_core_id();
  uint8_t * pIm2ColBase = pIm2ColBuffer + (2 * core_id * ch_in * dim_kernel_x * dim_kernel_y); // do not change this! :)
  int i_out_y, i_out_x, i_ker_y, i_ker_x;
  int Log2Core = log2(NUM_CORES);

  uint8_t flag_dim_out_x_odd = dim_out_x & 0x01;

  // int chunk = (dim_out_y >> Log2Core);

  int start_pixel = 0;         // min((chunk * core_id), dim_out_y);
  int stop_pixel  = dim_out_y; // min(start_pixel + chunk, dim_out_y);

  uint8_t *pIm2Col = pIm2ColBase;
  uint8_t *pOutBuffer = pOut + (start_pixel * ch_out_r * dim_out_x);

  // #pragma omp parallel for schedule(static)
  for (i_out_y = start_pixel; i_out_y < stop_pixel; i_out_y++)
  {
    
    /* MAIN UNROLLED LOOP */
    for(i_out_x=0; i_out_x<(dim_out_x); i_out_x++)
    {

      /* IM2COL TRANSFORM */
      if(i_out_y < padding_y_top)
      {
        for(i_ker_y=((i_out_y * stride_y) - padding_y_top); i_ker_y<((i_out_y * stride_y) - padding_y_top + dim_kernel_y); i_ker_y++)
        {
          for(i_ker_x=((i_out_x * stride_x) - padding_x_left); i_ker_x<((i_out_x * stride_x) - padding_x_left + dim_kernel_x); i_ker_x++)
          {
            if((i_ker_y < 0) || (i_ker_y >= dim_in_y) || (i_ker_x < 0) || (i_ker_x >= dim_in_x))
            {
              pulp_zero_mem(pIm2Col, ch_in);
            }
            else
            {
              pulp_nn_im2col_u8_to_u8((uint8_t*) (pIn + ((i_ker_y * dim_in_x + i_ker_x) * ch_in_r)), pIm2Col, ch_in);
            }
            pIm2Col+=ch_in;
          }
        }
      }
      else if(i_out_y < dim_out_y - padding_y_bottom)
      {
        if(i_out_x < padding_x_left)
        {
          for(i_ker_y=((i_out_y * stride_y) - padding_y_top); i_ker_y<((i_out_y * stride_y) - padding_y_top + dim_kernel_y); i_ker_y++)
          {
            for(i_ker_x=((i_out_x * stride_x) - padding_x_left); i_ker_x<((i_out_x * stride_x) - padding_x_left + dim_kernel_x); i_ker_x++)
            {
              if((i_ker_x < 0) || (i_ker_x >= dim_in_x))
              {
                pulp_zero_mem(pIm2Col, ch_in);
              }
              else
              {
                pulp_nn_im2col_u8_to_u8((uint8_t*) (pIn + ((i_ker_y * dim_in_x + i_ker_x) * ch_in_r)), pIm2Col, ch_in);
              }
              pIm2Col+=ch_in;
            }
          }
        }
        else if(i_out_x < (dim_out_x - padding_x_right))
        {
          for(i_ker_y=((i_out_y * stride_y) - padding_y_top); i_ker_y<((i_out_y * stride_y) - padding_y_top + dim_kernel_y); i_ker_y++)
          {
            pulp_nn_im2col_u8_to_u8((uint8_t*) pIn + (i_ker_y * dim_in_x + i_out_x * stride_x - padding_x_left)*ch_in_r,pIm2Col,ch_in * dim_kernel_x);
            pIm2Col+=(ch_in * dim_kernel_x);
          }
        }
        else
        {
          for(i_ker_y=((i_out_y * stride_y) - padding_y_top); i_ker_y<((i_out_y * stride_y) - padding_y_top + dim_kernel_y); i_ker_y++)
          {
            for(i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++)
            {
              if((i_ker_x < 0) || (i_ker_x >= dim_in_x))
              {
                pulp_zero_mem(pIm2Col, ch_in);
              }
              else
              {
                pulp_nn_im2col_u8_to_u8((uint8_t *)pIn + (i_ker_y*dim_in_x+i_ker_x)* ch_in_r, pIm2Col, ch_in);
              }
              pIm2Col+=ch_in;
            }
          }
        }
      }
      else
      {
        for(i_ker_y=((i_out_y * stride_y) - padding_y_top); i_ker_y<((i_out_y * stride_y) - padding_y_top + dim_kernel_y); i_ker_y++)
        {
          for(i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++)
          {
            if(i_ker_y < 0 || (i_ker_y >= dim_in_y) || i_ker_x < 0 || i_ker_x >= dim_in_x)
            {
              pulp_zero_mem(pIm2Col, ch_in);
            }
            else
            {
              pulp_nn_im2col_u8_to_u8((uint8_t *) pIn + (i_ker_y * dim_in_x + i_ker_x) * ch_in_r, pIm2Col, ch_in);
            }
            pIm2Col+=ch_in;
          }
        }
      }

      /* MATRIX MULTIPLICATION LOOP */
      if(pIm2Col == (pIm2ColBase + ((ch_in * dim_kernel_x * dim_kernel_y) << 1)))
      {
        // compute mat-mul and update output pointer
        pOutBuffer = pulp_nn_matmul_u8_i8(
          pIm2ColBase,
          pBias,
          pOutBuffer,
          pOutBuffer + ch_out_r,
          pWeight,
          pKappa,
          pLambda,
          out_mult,
          out_shift,
          (ch_in * dim_kernel_x * dim_kernel_y),
          ch_out,
          flag_relu,
          flag_batch_norm
        );
        // roll-back im2col pointer
        pIm2Col = pIm2ColBase;
      }
    }
    
    /* LEFTOVER */
    if(pIm2Col != pIm2ColBase)
    {
      printf("Should not enter here in the exercise!!\n");
      const int8_t *pA = pWeight;
      int i;
      int32_t * k1 = pKappa;
      int32_t * lambda1 = pLambda;
      for(i = 0; i < ch_out; i++)
      {
        int sum = 0;
        if (pBias != NULL)
        {
          sum = ((int) (*pBias++));
        }

        uint8_t *pB = pIm2ColBase;
        uint16_t col_cnt_im2col = ch_in * dim_kernel_x * dim_kernel_y >> 2;
        for(int j=0; j < col_cnt_im2col; j++)
        {
          v4s inA = *((v4s*) pA);
          v4u inB = *((v4u*) pB);

          sum = SumDotp4(inB, inA, sum);
          pA+=4;
          pB+=4;
        }
        col_cnt_im2col = (ch_in * dim_kernel_y * dim_kernel_x) & 0x3;
        while (col_cnt_im2col)
        {
          int8_t inA1 = *pA++;
          uint8_t inB1 = *pB++;
          asm volatile("": : :"memory");
          sum += inA1 * inB1;

          col_cnt_im2col--;
        }
        if (flag_batch_norm && flag_relu)
        {
          *pOutBuffer = pulp_nn_bn_quant_u8(sum, *k1, *lambda1, out_shift);
          k1++;
          lambda1++;
          pOutBuffer++;
        }
        else
        {
          if(flag_relu == 1)
          {
            *pOutBuffer = pulp_nn_quant_u8(sum, out_mult, out_shift);
            pOutBuffer++;
          }
          else
          {
            *pOutBuffer = (uint8_t) clip8(sum >> out_shift);
            pOutBuffer++;
          }
        }
      }
    }
    pIm2Col = pIm2ColBase;
  }
  pi_cl_team_barrier(0);
}
