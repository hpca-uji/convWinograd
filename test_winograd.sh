
VARIANT=WINGRD

NMIN=1
NMAX=1
NSTEP=1

KMIN=512
KMAX=512
KSTEP=64

CMIN=512
CMAX=512
CSTEP=64  

HMIN=4
HMAX=4
HSTEP=2   

WMIN=4
WMAX=4
WSTEP=2   

RMIN=3
RMAX=3
RSTEP=1   

SMIN=3 
SMAX=3 
SSTEP=1   

VPADMIN=1
VPADMAX=1
VPADSTEP=1

HPADMIN=1
HPADMAX=1
HPADSTEP=1   

VISUAL=0
TIMIN=5.0
TEST=T

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
