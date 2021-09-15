
export VARIANT=WINGRD

export NMIN=16
export NMAX=18
export NSTEP=1

export KMIN=16
export KMAX=18
export KSTEP=1

export CMIN=16
export CMAX=18
export CSTEP=1   

export HMIN=32
export HMAX=32
export HSTEP=1   

export WMIN=32
export WMAX=32
export WSTEP=1   

export RMIN=2
export RMAX=5
export RSTEP=1   

export SMIN=2 
export SMAX=5 
export SSTEP=1   

export VPADMIN=0
export VPADMAX=3
export VPADSTEP=1

export HPADMIN=0
export HPADMAX=3
export HPADSTEP=1   

VISUAL=0
TIMIN=0.0
TEST=T

OMP_SET_NUM_THREADS=20
./test_winograd.x \
  $VARIANT \
  $NMIN $NMAX $NSTEP \
  $KMIN $KMAX $KSTEP \
  $CMIN $CMAX $CSTEP \
  $HMIN $HMAX $HSTEP \
  $WMIN $WMAX $WSTEP \
  $RMIN $RMAX $RSTEP \
  $SMIN $SMAX $SSTEP \
  $VPADMIN $VPADMAX $VPADSTEP \
  $HPADMIN $HPADMAX $HPADSTEP \
  $VISUAL $TIMIN $TEST \
