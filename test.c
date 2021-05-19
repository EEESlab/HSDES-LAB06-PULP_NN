/*
 * test.c
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


#define CHECK

#define PERFORMANCE

#if (KERNEL == 888)
#define INPUT 8
#define OUTPUT 8
#define WEIGHTS 8
#include "GoldenModelConvolution/golden.h"
#include "DataAllocationConvolution/data_allocation.h"
#endif



void test();
void pulp_parallel();

void pulp_parallel()
{
  pi_cl_team_fork(NUM_CORES, (void *)test, NULL);
}

void test()
{
  uint32_t errors = 0;

  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;

  if(pi_core_id()==0)
  {
#ifdef PERFORMANCE
#ifdef VERBOSE_PERF
    printf("MACs=%d\n", DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT);
#endif /* VERBOSE */
#endif /* PERFORMANCE */
#if INPUT == 2
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2); i++)
    {
      uint8_t c_data = bitins(IN_INT2_L2[4*i], n_mask2, IN_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, IN_INT2_L2[(4*i)+2], mask4, off4);
      IN_INT8_L1[i] = bitins(c_data, n_mask6, IN_INT2_L2[(4*i)+3], mask6, off6);
    }
#elif INPUT == 4
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1); i++)
    {
      IN_INT8_L1[i] = bitins(IN_INT4_L2[2*i], n_mask, IN_INT4_L2[(2*i)+1], mask, off);
    }
#elif INPUT == 8
    for(int i=0; i<(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN); i++)
    {
      IN_INT8_L1[i] = IN_INT8_L2[i];
    }
#endif /* INPUT */
#if WEIGHTS == 2
    for(int i=0; i<((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT) >> 2); i++)
    {
      int8_t c_data = bitins(WEIGHT_INT2_L2[4*i], n_mask2, WEIGHT_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, WEIGHT_INT2_L2[(4*i)+2], mask4, off4);
      WEIGHT_INT8_L1[i] = bitins(c_data, n_mask6, WEIGHT_INT2_L2[(4*i)+3], mask6, off6);
    }
#elif WEIGHTS == 4
    for(int i=0; i<((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT) >> 1); i++)
    {
      WEIGHT_INT8_L1[i] = bitins(WEIGHT_INT4_L2[2*i], n_mask, WEIGHT_INT4_L2[(2*i)+1], mask, off);
    }
#elif WEIGHTS == 8
    for(int i=0; i<(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT); i++)
    {
      WEIGHT_INT8_L1[i] = WEIGHT_INT8_L2[i];
    }
#endif /* WEIGHTS */
    for(int i=0; i<CH_IM_OUT; i++)
    {
      KAPPA_L1[i] = KAPPA_L2[i];
    }
    for(int i=0; i<CH_IM_OUT; i++)
    {
      LAMBDA_L1[i] = LAMBDA_L2[i];
    }
  }
  pi_cl_team_barrier(0);

#ifdef PERFORMANCE
#ifdef VERBOSE_PERF
for (int k=0; k < 13; k++)
{
  if ( (k>=3))
  {         
    if(k==3) pi_perf_conf(1<<PI_PERF_CYCLES);     
    if(k==4) pi_perf_conf(1<<PI_PERF_IMISS);      
    if(k==5) pi_perf_conf(1<<PI_PERF_LD_EXT);   
    if(k==6) pi_perf_conf(1<<PI_PERF_ST_EXT);    
    if(k==7) pi_perf_conf(1<<PI_PERF_TCDM_CONT);   
    if(k==8) pi_perf_conf(1<<PI_PERF_INSTR);     
    if(k==9) pi_perf_conf(1<<PI_PERF_ACTIVE_CYCLES);    
    if(k==10) pi_perf_conf(1<<PI_PERF_LD_STALL);    
    if(k==11) pi_perf_conf(1<<PI_PERF_JR_STALL);   
    if(k==12) pi_perf_conf(1<<PI_PERF_BRANCH);   
    pi_perf_reset();            
    pi_perf_stop();           
    pi_perf_start(); 
  }
#else
  pi_perf_conf(1<<PI_PERF_CYCLES);          
  pi_perf_reset();                      
  pi_perf_stop();                       
  pi_perf_start(); 
#endif /* VERBOSE */
#endif /* PERFORMANCE */


#if (KERNEL == 888)
pulp_nn_conv_u8_u8_i8(IN_INT8_L1,
                    IM2COL_L1,
                    BIAS_L1,
                    OUT_L1,
                    WEIGHT_INT8_L1,
                    KAPPA_L1,
                    LAMBDA_L1,
                    OUT_MULT,
                    OUT_SHIFT,
                    DIM_IM_IN_X,
                    DIM_IM_IN_Y,
                    CH_IM_IN,
                    DIM_IM_OUT_X,
                    DIM_IM_OUT_Y,
                    CH_IM_OUT,
                    DIM_KERNEL_X,
                    DIM_KERNEL_Y,
                    PADDING_Y_TOP,
                    PADDING_Y_BOTTOM,
                    PADDING_X_LEFT,
                    PADDING_X_RIGHT,
                    STRIDE_X,
                    STRIDE_Y,
                    1,
                    1
                        );
#endif



#ifdef PERFORMANCE
#ifdef VERBOSE_PERF
  if( (k>=3 ))
  {
    pi_perf_stop();      
    int cid = pi_core_id();   
    if(k==3) printf("[%d] : num_cycles: %d\n",cid,pi_perf_read(PI_PERF_CYCLES) ); 
    if(k==4) printf("[%d] : num_instr_miss: %d\n",cid,pi_perf_read(PI_PERF_IMISS) ); 
    if(k==5) printf("[%d] : num_ext_load: %d\n",cid,pi_perf_read(PI_PERF_LD_EXT) ); 
    if(k==6) printf("[%d] : num_ext_Store: %d\n",cid,pi_perf_read(PI_PERF_ST_EXT) ); 
    if(k==7) printf("[%d] : num_tcdm_contentions: %d\n",cid,pi_perf_read(PI_PERF_TCDM_CONT) ); 
    if(k==8) printf("[%d] : num_instrs: %d\n",cid,pi_perf_read(PI_PERF_INSTR) ); 
    if(k==9) printf("[%d] : num_active_cycles: %d\n",cid,pi_perf_read(PI_PERF_ACTIVE_CYCLES) ); 
    if(k==10) printf("[%d] : num_load_stalls: %d\n",cid,pi_perf_read(PI_PERF_LD_STALL ) ); 
    if(k==11) printf("[%d] : num_jumpr_stalls: %d\n",cid,pi_perf_read(PI_PERF_JR_STALL ) ); 
    if(k==12) printf("[%d] : num_branch: %d\n",cid,pi_perf_read(PI_PERF_BRANCH ) ); 
   }             
 }
#else
  pi_perf_stop();          
  int cid = pi_core_id();   
  int perf_cyc =  pi_perf_read(PI_PERF_CYCLES);
  int MACs = DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT;
  float perf_MAC =  (float)MACs/perf_cyc;
  if (cid == 0)
  {
    printf("\n[%d] : num_cycles: %d\n",cid,perf_cyc); 
    printf("[%d] : MACs: %d\n",cid,MACs ); 
    printf("[%d] : MAC/cycle: %f\n",cid,perf_MAC ); 
    printf("[%d] : n. of Cores: %d\n",cid,NUM_CORES); 
  }
#endif /* VERBOSE */
  pi_cl_team_barrier(0);
#endif /* PERFORMANCE */
#ifdef CHECK
  if(pi_core_id()==0)
  {
#if OUTPUT == 2
    for (int i=0; i<((DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<((DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2); i++)
    {
      uint8_t c_data = bitins(OUT_INT2_L2[4*i], n_mask2, OUT_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, OUT_INT2_L2[(4*i)+2], mask4, off4);
      OUT_INT8_L2[i] = bitins(c_data, n_mask6, OUT_INT2_L2[(4*i)+3], mask6, off6);
#elif OUTPUT == 4
    for (int i=0; i<((DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<((DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1); i++)
    {
      OUT_INT8_L2[i] = bitins(OUT_INT4_L2[2*i], n_mask, OUT_INT4_L2[(2*i)+1], mask, off);
#elif OUTPUT == 8
    for (int i=0; i<(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT); i++)
    {
#endif /* OUTPUT */
      if(OUT_L2[i] != OUT_INT8_L2[i])
      {
#ifdef VERBOSE_CHECK
        printf("error at index %d, %d instead of %d\n", i, OUT_L2[i], OUT_INT8_L2[i]);
#endif /* VERBOSE */
        errors++;
      }
    }
    printf("errors: %d\n", errors);
  }
  pi_cl_team_barrier(0);
#endif /* CHECK */
}

///////////////////////////////////////////////////////////////////
////------------------------MAIN------------------------------/////
///////////////////////////////////////////////////////////////////

int main()
{
  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};
  // task parameters allocation
  pi_cluster_task(&cluster_task, pulp_parallel, NULL);
  cluster_task.stack_size = 1024;
  cluster_task.slave_stack_size = 1024;
  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id=0;
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return -1;
  // Then offload an entry point, this will get executed on the cluster controller
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  // closing of the cluster
  pi_cluster_close(&cluster_dev);

  return 0;
}
