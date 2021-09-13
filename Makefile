#-----------------------------------

  CHECK_PARAMS = 
# CHECK_PARAMS = -DCHECK

#-----------------------------------

CC       = gcc
CLINKER  = gcc
CFLAGS   = -fopenmp -ftree-vectorize -O3
aOBJS    =

UNAME    = $(shell uname -m)

ifeq ($(UNAME), x86_64) 
  ifdef MKLROOT
    FLAGS    = -DMKL -DEXTERN_CBLAS -mavx -Wall
    #OPTFLAGS = -L${MKLROOT}/lib/intel64 -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl
    OPTFLAGS = -L${MKLROOT}/lib/intel64_lin -Wl,--no-as-needed -lmkl_avx512 -lmkl_def -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
    #OPTFLAGS = -L${MKLROOT}/lib/intel64_lin -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
    INCLUDE  = -m64  -I"${MKLROOT}/include"
    OBJS    += conv_winograd_nchw_fp32.o \
               conv_winograd_3x3_2x2_nchw_avx_fp32.o \
               conv_winograd_2x2_3x3_nchw_avx_fp32.o \
               conv_winograd_4x4_3x3_nchw_avx_fp32.o \
               conv_winograd_2x2_5x5_nchw_avx_fp32.o
  else
    DTYPE    = -DFP32 
    OBJS    += conv_winograd_nchw_fp32.o gemm.o
  endif
else ifeq ($(UNAME), aarch64)
    FLAGS    = -DARM_NEON -DEXTERN_CBLAS
    OPTFLAGS = -lblas
    OBJS    += conv_winograd_3x3_2x2_nchw_neon_fp32.o \
               conv_winograd_2x2_3x3_nchw_neon_fp32.o \
               conv_winograd_4x4_3x3_nchw_neon_fp32.o \
               conv_winograd_2x2_5x5_nchw_neon_fp32.o
endif


LIBCONVWINOGRAD = libconvwinograd.so

#-----------------------------------

default: $(LIBCONVWINOGRAD)

$(LIBCONVWINOGRAD): $(OBJS)
	$(CLINKER) $(OBJS) $(OPTFLAGS) -shared -o $@

#-----------------------------------

.c.o:
	$(CC) $(CFLAGS) $(FLAGS) $(INCLUDE) $(DTYPE) -c $*.c

#-----------------------------------

clean:
	rm -f *.o $(LIBCONVWINOGRAD) 

#-----------------------------------


