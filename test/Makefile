APP = test

APP_SRCS = test.c

ifndef kernel
kernel=888
else
kernel = $(kernel)
endif

APP_SRCS += src/Convolution/pulp_nn_conv_u8_u8_i8.c

APP_SRCS += src/MatrixMultiplication/pulp_nn_matmul_u8_i8.c

CONFIG_OPENMP = 1

ifndef cores
cores=1
else
cores = $(cores)
endif

ifeq ($(perf), 1)
APP_CFLAGS += -DVERBOSE_PERF
endif

ifeq ($(check), 1)
APP_CFLAGS += -DVERBOSE_CHECK
endif

APP_CFLAGS += -O3 -Iinclude -w -flto
APP_CFLAGS += -DNUM_CORES=$(cores) -DKERNEL=$(kernel)

APP_LDFLAGS += -lm -flto


include $(RULES_DIR)/pmsis_rules.mk
