#-----------------------------------

  CHECK_PARAMS = 
# CHECK_PARAMS = -DCHECK

#-----------------------------------

CC       = gcc
CLINKER  = gcc
CFLAGS   = -fopenmp -ftree-vectorize -O3
OBJS     = libconvwinograd.o

UNAME    = $(shell uname -m)

ifeq ($(UNAME), x86_64) 
  ifdef MKLROOT
    FLAGS    = -DMKL -DEXTERN_CBLAS
    OPTFLAGS = -L${MKLROOT}/lib/intel64 -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl
    INCLUDE  = -m64  -I"${MKLROOT}/include"
  else
    OBJS    += gemm.o
    DTYPE    = -DFP32 
  endif
else ifeq ($(UNAME), aarch64)
    FLAGS    = -DARM_NEON -DEXTERN_CBLAS
    OPTFLAGS = -lblas
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
	rm *.o $(LIBCONVWINOGRAD) 

#-----------------------------------


