#-----------------------------------

  CHECK_PARAMS = 
# CHECK_PARAMS = -DCHECK

#-----------------------------------

CC       = gcc
CLINKER  = gcc
CFLAGS   = -fopenmp -ftree-vectorize -O3 -DFP32
# OBJS    =

UNAME    = $(shell uname -m)

ifeq ($(UNAME), x86_64) 
  ifdef MKLROOT
    FLAGS    = -DMKL -DEXTERN_CBLAS -mavx
    OPTFLAGS = -L${MKLROOT}/lib/intel64_lin -Wl,--no-as-needed -lmkl_avx512 -lmkl_def -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
    #OPTFLAGS = -L${MKLROOT}/lib/intel64_lin -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
    INCLUDE  = -m64  -I"${MKLROOT}/include"
    OBJS    += conv_winograd_3x3_2x2_sse_fp32_nchw.o \
               conv_winograd_2x2_3x3_sse_fp32_nchw.o \
               conv_winograd_4x4_3x3_sse_fp32_nchw.o \
               conv_winograd_2x2_5x5_sse_fp32_nchw.o \
               conv_winograd_3x3_2x2_sse_fp32_nhwc.o \
               conv_winograd_2x2_3x3_sse_fp32_nhwc.o \
               conv_winograd_4x4_3x3_sse_fp32_nhwc.o \
               conv_winograd_2x2_5x5_sse_fp32_nhwc.o \
               conv_winograd_fp32_nchw.o
  else
    OPTFLAGS = -lm -lgomp
    OBJS    += conv_winograd_nchw_fp32.o gemm.o
  endif
else ifeq ($(UNAME), aarch64)
    FLAGS    = -DARM_NEON -DEXTERN_CBLAS
    OPTFLAGS = -L/home/dolzm/install/blis/lib -lblis -lgomp -lm
    OBJS    += conv_winograd_3x3_2x2_neon_fp32_nchw.o \
               conv_winograd_2x2_3x3_neon_fp32_nchw.o \
               conv_winograd_4x4_3x3_neon_fp32_nchw.o \
               conv_winograd_2x2_5x5_neon_fp32_nchw.o \
               conv_winograd_3x3_2x2_neon_fp32_nhwc.o \
               conv_winograd_2x2_3x3_neon_fp32_nhwc.o \
               conv_winograd_4x4_3x3_neon_fp32_nhwc.o \
               conv_winograd_2x2_5x5_neon_fp32_nhwc.o \
               conv_winograd_fp32_nchw.o
endif


LIBCONVWINOGRAD = libconvwinograd.so
WINOGRADDRIVER = test_winograd.x

default: $(LIBCONVWINOGRAD) $(WINOGRADDRIVER)

#-----------------------------------

$(LIBCONVWINOGRAD): $(OBJS)
	$(CLINKER) $(OBJS) $(OPTFLAGS) -shared -o $@

# #-----------------------------------

$(WINOGRADDRIVER): test_winograd.c $(LIBCONVWINOGRAD) sutils.o
	$(CC) -DFP32 $^ $(OPTFLAGS) -o $@

# #-----------------------------------

%_nchw.o: %.c
	$(CC) $(CFLAGS) -DTENSOR_FORMAT_NCHW $(FLAGS) $(INCLUDE) $(DTYPE) -c $*.c -o $@

# #-----------------------------------

%_nhwc.o: %.c
	$(CC) $(CFLAGS) -DTENSOR_FORMAT_NHWC $(FLAGS) $(INCLUDE) $(DTYPE) -c $*.c -o $@

# #-----------------------------------

clean:
	rm -f *.o $(LIBCONVWINOGRAD) $(WINOGRADDRIVER)

#-----------------------------------


