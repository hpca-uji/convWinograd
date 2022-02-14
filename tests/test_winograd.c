/* 
   WINOGRAD 

   -----

   WINOGRAD is an implementation of the Winograd-based convolution transform

   -----

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
   You should have received a copy of the GNU General Public License along with
   this program. If not, see <http://www.gnu.org/licenses/>.

   -----

   author    = "Enrique S. Quintana-Orti"
   contact   = "quintana@disca.upv.es"
   copyright = "Copyright 2021, Universitat Politecnica de Valencia"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#define DTYPE float

#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )
#define min(a,b)     ( (a) > (b) ? (b) : (a) )

#define Drow_nchw(a1,a2,a3,a4)  D[ (a1)*(ldD1)+(a2)*(ldD2)+(a3)*(ldD3)+(a4) ]
#define Frow_nchw(a1,a2,a3,a4)  F[ (a1)*(ldF1)+(a2)*(ldF2)+(a3)*(ldF3)+(a4) ]
#define Yrow_nchw(a1,a2,a3,a4)  Y[ (a1)*(ldY1)+(a2)*(ldY2)+(a3)*(ldY3)+(a4) ]
#define Ygrow_nchw(a1,a2,a3,a4) Yg[ (a1)*(ldY1)+(a2)*(ldY2)+(a3)*(ldY3)+(a4) ]

#define Drow_nhwc(a1,a2,a3,a4)  D[ (a1)*(ldD1)+(a3)*(ldD2)+(a4)*(ldD3)+(a2) ]
#define Frow_nhwc(a1,a2,a3,a4)  F[ (a2)*(ldF1)+(a3)*(ldF2)+(a4)*(ldF3)+(a1) ]
#define Yrow_nhwc(a1,a2,a3,a4)  Y[ (a1)*(ldY1)+(a3)*(ldY2)+(a4)*(ldY3)+(a2) ]
#define Ygrow_nhwc(a1,a2,a3,a4) Yg[ (a1)*(ldY1)+(a3)*(ldY2)+(a4)*(ldY3)+(a2) ]

extern int    print_tensor4D( char *, int, int, int, int, DTYPE *, int, int, int );
extern int    print_matrix( char *, char, int, int, DTYPE *, int );
extern int    generate_tensor4D( int, int, int, int, DTYPE *, int, int, int );
extern double dclock();

#if __x86_64__ && __LP64__
#define VARIANT sse
#elif __aarch64__ && __LP64__
#define VARIANT neon
#elif __riscv && __riscv_xlen==64
#define VARIANT riscv
#endif

#define CONV_ARGS    int m, int r, int n, int k, int c, \
                     int hi, int wi, int kh, int kw, \
                     int vpadding, int hpadding, \
                     float *D, int ldD1, int ldD2, int ldD3, \
                     float *F, int ldF1, int ldF2, int ldF3, \
                     float *Y, int ldY1, int ldY2, int ldY3, \
                     float *biases, float *Bt, float *G, float *At,\
                     float *U,  float *V, float *M, \
                     const char relu, const char bn, \
                     float *running_mean, float *inv_std, \
                     float *gamma, float *beta

#define CONV_PARAMS(v) m, r, n, k, c, \
                     h, w, r, s, \
                     vpadding, hpadding, \
                     D,  ldD1, ldD2, ldD3, \
                     F,  ldF1, ldF2, ldF3, \
                     Y, ldY1, ldY2, ldY3, \
                     NULL, Bt_ ## v , G_ ## v , At_ ## v, U,  V, M, \
                    'F', 'F', NULL, NULL, NULL, NULL
//                     NULL, NULL, NULL, NULL, U,  V, M, \

#ifdef VARIANT
#define DECL_FUNC2(v, a, f) conv_winograd_ ## v ## _ ## a ## _fp32_ ## f (CONV_ARGS)
#define DECL_FUNC(v, a, f) DECL_FUNC2(v, a, f)
#define CALL_FUNC2(v, a, f) conv_winograd_ ## v ## _ ## a ## _fp32_ ## f (CONV_PARAMS(v))
#define CALL_FUNC(v, a, f) CALL_FUNC2(v, a, f)
extern void DECL_FUNC(3x3_2x2, VARIANT, nchw);
extern void DECL_FUNC(2x2_3x3, VARIANT, nchw);
extern void DECL_FUNC(4x4_3x3, VARIANT, nchw);
extern void DECL_FUNC(2x2_5x5, VARIANT, nchw);
extern void DECL_FUNC(3x3_2x2, VARIANT, nhwc);
extern void DECL_FUNC(2x2_3x3, VARIANT, nhwc);
extern void DECL_FUNC(4x4_3x3, VARIANT, nhwc);
extern void DECL_FUNC(2x2_5x5, VARIANT, nhwc);
#else
#define CALL_FUNC2(v, a, f) conv_winograd_fp32_ ## f ## (CONV_PARAMS(v))
#define CALL_FUNC(v, a, f) CALL_FUNC2(v, a, f)
extern void conv_winograd_fp32(CONV_ARGS);
#endif

#define NCHW 0
#define NHWC 1

void convDirect( int, int, int, 
                 int, int, 
                 int, int, 
                 int, int, 
                 DTYPE *, int, int, int, 
                 DTYPE *, int, int, int, 
                 DTYPE *, int, int, int, int );

int main(int argc, char *argv[])
{
  char  test;
  char* variant;
  DTYPE *D, *F, *Y, *Yg, *U, *V, *M;
  double t1, t2, time, tmin, error, nrm, tmp, errorthd, flops, GFLOPS;
  int    m, t, tmin_,
         nmin,  nmax,  nstep,
         kmin,  kmax,  kstep,
         cmin,  cmax,  cstep,
         hmin,  hmax,  hstep,
         wmin,  wmax,  wstep,
         rmin,  rmax,  rstep,
         smin,  smax,  sstep,
         prmax, psmax, ret,
         tformat, tformatmin, tformatmax,
         n, k, c,
         h, w,
         r, s,
         pr, ps,
         in, ir, is, ic, ik, ih, iw,
         ldD1, ldD2, ldD3,
         ldF1, ldF2, ldF3,
         ldY1, ldY2, ldY3,
         visual, nreps,
         tile_H, tile_W, ho, wo, homax, womax,
         vpadding, vpaddingmin, vpaddingmax, vpaddingstep,
         hpadding, hpaddingmin, hpaddingmax, hpaddingstep;

  // The definition of these matrices are not necessary as the vectorized
  // versions implicitly contain them in the corresponding codes
  // This is only necessary for the generic version based on gemm operations
  // These parameteres for the vectorized variants can be NULL

  /*** WINOGRAD 3x3 2x2 ***/
  DTYPE Bt_3x3_2x2[16] = {1.0,  0.0, -1.0,  0.0, 
                          0.0,  1.0,  1.0,  0.0, 
                          0.0, -1.0,  1.0,  0.0, 
                          0.0, -1.0,  0.0,  1.0};

  DTYPE G_3x3_2x2[8]   = {1.0,  0.0, 
                          0.5,  0.5, 
                          0.5, -0.5, 
                          0.0,  1.0};

  DTYPE At_3x3_2x2[12] = {1.0, 1.0,  1.0,  0.0, 
                          0.0, 1.0, -1.0,  0.0,
                          0.0, 1.0,  1.0,  1.0};

  /*** WINOGRAD 2x2 3x3 ***/
  DTYPE Bt_2x2_3x3[16] = {1.0,  0.0, -1.0,  0.0,
                          0.0,  1.0,  1.0,  0.0,
                          0.0, -1.0,  1.0,  0.0,
                          0.0,  1.0,  0.0, -1.0};

  DTYPE G_2x2_3x3[12]  = {1.0,  0.0, 0.0,
                          0.5,  0.5, 0.5, 
                          0.5, -0.5, 0.5,
                          0.0,  0.0, 1.0};

  DTYPE At_2x2_3x3[8]  = {1.0, 1.0,  1.0,  0.0, 
                          0.0, 1.0, -1.0, -1.0};

  /*** WINOGRAD 4x4 3x3 ***/
  DTYPE Bt_4x4_3x3[36] = {4.0,  0.0, -5.0,  0.0,  1.0,  0.0,
                          0.0, -4.0, -4.0,  1.0,  1.0,  0.0,
                          0.0,  4.0, -4.0, -1.0,  1.0,  0.0,
                          0.0, -2.0, -1.0,  2.0,  1.0,  0.0,
                          0.0,  2.0, -1.0, -2.0,  1.0,  0.0,
                          0.0,  4.0,  0.0, -5.0,  0.0,  1.0};

  DTYPE G_4x4_3x3[18] =  {1.0/4.0,       0.0,      0.0,
                         -1.0/6.0,  -1.0/6.0, -1.0/6.0,
                         -1.0/6.0,   1.0/6.0, -1.0/6.0,
                         1.0/24.0,  1.0/12.0,  1.0/6.0,
                         1.0/24.0, -1.0/12.0,  1.0/6.0,
                              0.0,       0.0,      1.0};

  DTYPE At_4x4_3x3[24] = {1.0,  1.0,  1.0,  1.0,  1.0,  0.0,
                          0.0,  1.0, -1.0,  2.0, -2.0,  0.0,
                          0.0,  1.0,  1.0,  4.0,  4.0,  0.0,
                          0.0,  1.0, -1.0,  8.0, -8.0,  1.0};

  /*** WINOGRAD 2x2 5x5 ***/
  DTYPE Bt_2x2_5x5[36] = {4.0,  0.0, -5.0,  0.0,  1.0,  0.0,
                          0.0, -4.0, -4.0,  1.0,  1.0,  0.0,
                          0.0,  4.0, -4.0, -1.0,  1.0,  0.0,
                          0.0, -2.0, -1.0,  2.0,  1.0,  0.0,
                          0.0,  2.0, -1.0, -2.0,  1.0,  0.0,
                          0.0,  4.0,  0.0, -5.0,  0.0,  1.0};

  DTYPE G_2x2_5x5[30] =  {1.0/4.0,       0.0,        0.0,       0.0,      0.0,
                         -1.0/6.0,  -1.0/6.0,   -1.0/6.0,  -1.0/6.0, -1.0/6.0,
                         -1.0/6.0,   1.0/6.0,   -1.0/6.0,   1.0/6.0, -1.0/6.0,
                         1.0/24.0,  1.0/12.0,    1.0/6.0,   1.0/3.0,  2.0/3.0,
                         1.0/24.0, -1.0/12.0,    1.0/6.0,  -1.0/3.0,  2.0/3.0,
                              0.0,       0.0,        0.0,       0.0,      1.0};

  DTYPE At_2x2_5x5[12] = {1.0, 1.0,  1.0,  1.0,  1.0,  0.0,
                          0.0, 1.0, -1.0,  2.0, -2.0,  1.0};

  printf("# Program starts...\n");

  // printf("# -->Read data\n"); fflush(stdout);
  variant = argv[1];
  nmin  = atoi(argv[2]);
  nmax  = atoi(argv[3]);
  nstep = atoi(argv[4]);

  kmin  = atoi(argv[5]);
  kmax  = atoi(argv[6]);
  kstep = atoi(argv[7]);

  cmin  = atoi(argv[8]);
  cmax  = atoi(argv[9]);
  cstep = atoi(argv[10]);

  hmin  = atoi(argv[11]);
  hmax  = atoi(argv[12]);
  hstep = atoi(argv[13]);

  wmin  = atoi(argv[14]);
  wmax  = atoi(argv[15]);
  wstep = atoi(argv[16]);

  rmin  = atoi(argv[17]);
  rmax  = atoi(argv[18]);
  rstep = atoi(argv[19]);

  smin  = atoi(argv[20]);
  smax  = atoi(argv[21]);
  sstep = atoi(argv[22]);

  vpaddingmin  = atoi(argv[23]);
  vpaddingmax  = atoi(argv[24]);
  vpaddingstep = atoi(argv[25]);

  hpaddingmin  = atoi(argv[26]);
  hpaddingmax  = atoi(argv[27]);
  hpaddingstep = atoi(argv[28]);

  visual = atoi(argv[29]);
  tmin   = atof(argv[30]);
  test   = argv[31][0];

  printf("# =======================================================================================================");
  if ( test=='T' ) printf("======="); printf("\n");
  printf("# Driver for the evaluation of Winograd\n");
  printf("# =======================================================================================================");
  if ( test=='T' ) printf("======="); printf("\n");
  printf("#    variant     n     k     c     h     w    kh    kw  vpad  hpad  format       Time    GFLOPS     Error");
  if ( test=='T' ) printf(" Status"); printf("\n");
  
  // Allocate space for data 
  // printf("# -->Allocate data\n"); fflush(stdout);
  tformatmin=0; tformatmax=2;
  m = 2; t = 6;
  tmin_ = 4;

  homax = floor(((double) hmax + 2 * vpaddingmax - rmin) / 1) + 1;
  womax = floor(((double) wmax + 2 * hpaddingmax - smin) / 1) + 1;

  D = (DTYPE *) malloc( nmax*cmax*hmax*wmax*sizeof(DTYPE)); 
  F = (DTYPE *) malloc( kmax*cmax*rmax*smax*sizeof(DTYPE));   
  Y = (DTYPE *) malloc( nmax*kmax*homax*womax*sizeof(DTYPE));   

  tile_H = ceil(((double) hmax + 2 * vpaddingmax - tmin_) / m) + 1;
  tile_W = ceil(((double) wmax + 2 * hpaddingmax - tmin_) / m) + 1;

  U  = (DTYPE *) malloc(t*t*kmax*cmax*sizeof(DTYPE));
  V  = (DTYPE *) malloc(t*t*cmax*(nmax * tile_H * tile_W)*sizeof(DTYPE));
  M  = (DTYPE *) malloc(t*t*kmax*(nmax * tile_H * tile_W)*sizeof(DTYPE));

  if ( test=='T' )
    Yg = (DTYPE *) malloc( nmax*kmax*homax*womax*sizeof(DTYPE) );   

#if defined(FP16)
  errorthd = 1.0e-3;
#elif defined(FP32)
  errorthd = 1.0e-6;
#elif defined(FP64)
  errorthd = 1.0e-14;
#endif

  for ( n=nmin; n<=nmax; n+=nstep ){
  for ( k=kmin; k<=kmax; k+=kstep ){
  for ( c=cmin; c<=cmax; c+=cstep ){
  for ( h=hmin; h<=hmax; h+=hstep ){
  for ( w=wmin; w<=wmax; w+=wstep ){
  for ( r=rmin; r<=rmax; r+=rstep ){
  // for ( s=smin; s<=smax; s+=sstep ){
  for ( vpadding=vpaddingmin; vpadding<=vpaddingmax; vpadding+=vpaddingstep ){
  for ( hpadding=hpaddingmin; hpadding<=hpaddingmax; hpadding+=hpaddingstep ){
  for ( tformat=tformatmin; tformat<tformatmax; tformat+=1 ){
    s = r;
    //hpadding = vpadding;
    // Generate random data
    // printf("# -->Generate data\n"); fflush(stdout);
    ho = floor(((double) h + 2 * vpadding - r) / 1) + 1;
    wo = floor(((double) w + 2 * hpadding - s) / 1) + 1;

    if ( tformat == NCHW ) {
      // NCHW
      ldD3 = w;
      ldD2 = h*ldD3;
      ldD1 = c*ldD2;

      ldF3 = s;
      ldF2 = r*ldF3;
      ldF1 = c*ldF2;

      ldY3 = wo;
      ldY2 = ho*ldY3;
      ldY1 = k*ldY2;

      generate_tensor4D( n, c, h, w, D, ldD1, ldD2, ldD3 );
      generate_tensor4D( k, c, r, s, F, ldF1, ldF2, ldF3 );
    } else {
      // NHWC
      ldD3 = c;
      ldD2 = w*ldD3;
      ldD1 = h*ldD2;

      ldF3 = k;
      ldF2 = s*ldF3;
      ldF1 = r*ldF2;

      ldY3 = k;
      ldY2 = wo*ldY3;
      ldY1 = ho*ldY2;

      generate_tensor4D( n, h, w, c, D, ldD1, ldD2, ldD3 );
      generate_tensor4D( c, r, s, k, F, ldF1, ldF2, ldF3 );
   }

    // Print data
    if ( visual == 1 ){
      if ( tformat == NCHW ) {
        print_tensor4D( "D", n, c, h, w, D, ldD1, ldD2, ldD3 );
        print_tensor4D( "F", k, c, r, s, F, ldF1, ldF2, ldF3 );
      } else {
        print_tensor4D( "D", n, h, w, c, D, ldD1, ldD2, ldD3 );
        print_tensor4D( "F", c, r, s, k, F, ldF1, ldF2, ldF3 );
      }
    }

    // Set result to zeros
    for ( in=0; in<n; in++ )
    for ( ik=0; ik<k; ik++ )
    for ( ih=0; ih<ho; ih++ )
    for ( iw=0; iw<wo; iw++ ) {
        if ( tformat == NCHW ) Yrow_nchw(in,ik,ih,iw) = 0.0 ;
        else                   Yrow_nhwc(in,ik,ih,iw) = 0.0 ;
        if ( test=='T' ) {
          if ( tformat == NCHW ) Ygrow_nchw(in,ik,ih,iw) = 0.0 ;
          else                   Ygrow_nhwc(in,ik,ih,iw) = 0.0 ;
        }
    }

    // printf("# -->Solve problem\n"); fflush(stdout);

    time  = 0.0; 
    t1    = dclock();
    nreps = 0;
    while ( time <= tmin ) {
      // Winograd
      if ( strcmp(variant,"WINGRD\0")==0 ) {
        if (r == 2 && s == 2){
           m = 3;
           tformat == NCHW ?
             CALL_FUNC(3x3_2x2, VARIANT, nchw) :
             CALL_FUNC(3x3_2x2, VARIANT, nhwc);
        }
        else if (r == 3 && s == 3) {
           m = 2;
           tformat == NCHW ?
             CALL_FUNC(2x2_3x3, VARIANT, nchw) :
             CALL_FUNC(2x2_3x3, VARIANT, nhwc);
         //  m = 4;
         //  tformat == NCHW ?
         //    CALL_FUNC(4x4_3x3, VARIANT, nchw) :
         //    CALL_FUNC(4x4_3x3, VARIANT, nhwc);
        }
        else if (r == 5 && s == 5) {
           m = 2;
           tformat == NCHW ?
             CALL_FUNC(2x2_5x5, VARIANT, nchw) :
             CALL_FUNC(2x2_5x5, VARIANT, nhwc);
        }
        else break;
      }
      else {
        printf("Error: Unknown variant %s\n", variant);
        exit(-1);
      }
      nreps++;
         
      t2   = dclock();
      time = ( t2 > t1 ? t2 - t1 : 0.0 );
    }
    time = time/nreps;
    if ( nreps == 0 ) continue; 

    // Test result
    if ( test=='T' ) {
      convDirect( n, k, c, 
                  h, w, 
                  r, s, 
                  vpadding, hpadding, 
                  D,  ldD1, ldD2, ldD3, 
                  F,  ldF1, ldF2, ldF3, 
                  Yg, ldY1, ldY2, ldY3,
                  tformat );

      error = 0.0;
      nrm   = 0.0;
      for ( in=0; in<n; in++ )
      for ( ik=0; ik<k; ik++ )
      for ( ih=0; ih<ho; ih++ )
      for ( iw=0; iw<wo; iw++ ) {
        if ( tformat == NCHW ) {
          tmp = (double) Ygrow_nchw(in,ik,ih,iw);
          nrm += tmp*tmp;
          tmp = (double) dabs(Yrow_nchw(in,ik,ih,iw)-Ygrow_nchw(in,ik,ih,iw));
          error += tmp*tmp;
        } else {
          tmp = (double) Ygrow_nhwc(in,ik,ih,iw);
          nrm += tmp*tmp;
          tmp = (double) dabs(Yrow_nhwc(in,ik,ih,iw)-Ygrow_nhwc(in,ik,ih,iw));
          error += tmp*tmp;
        }
      }
      if ( nrm!=0.0 )
        error = sqrt(error) / sqrt(nrm);
      else
        error = sqrt(error);
    }
    else
      error = -1.0;

    // Print results
    if ( visual == 1 ) {
      if ( tformat == NCHW ) {
        print_tensor4D( "Yc", n, k, h, w, Y, ldY1, ldY2, ldY3 );
        print_tensor4D( "Ycd", n, k, h, w, Yg, ldY1, ldY2, ldY3 );
      } else {
        print_tensor4D( "Yc", n, h, w, k, Y, ldY1, ldY2, ldY3 );
        print_tensor4D( "Ycd", n, h, w, k, Yg, ldY1, ldY2, ldY3 );
      }
    }

    //printf("-->Results\n");
    //printf("   Time         = %12.6e seg.\n", time  );
    flops   = 2.0 * n * k * c * h * w * r * s;
    GFLOPS  = flops / (1.0e+9 * time );
    //printf("   GFLOPs       = %12.6e     \n", GFLOPS  );
    printf("      %6s %5d %5d %5d %5d %5d %5d %5d %5d %5d %7s %10.2e %9.2e %9.2e",
                    variant, n, k, c, h, w, r, s, vpadding, hpadding, ( tformat == NCHW ) ? "NCHW" : "NHWC", time, GFLOPS, error );
    if ( error<errorthd )   
      printf("   [OK]");   
    else 
      printf(" ******");   
    printf("\n");

  } } } } } } } } }
  /* Free data */
  free(Y);
  free(D);
  free(F);
  free(U);
  free(V);
  free(M);

  if ( test=='T' )
    free(Yg);
  printf("# End of program...\n");
  printf("# ================================================================================");
  if ( test=='T' ) printf("=======");   printf("\n");

  return 0;
}

void convDirect( int n, int k, int c, 
                 int h, int w, 
                 int r, int s, 
                 int vpadding, int hpadding, 
                 DTYPE *D, int ldD1, int ldD2, int ldD3,
	         DTYPE *F, int ldF1, int ldF2, int ldF3,
                 DTYPE *Yg, int ldY1, int ldY2, int ldY3,
                 int tformat )
{ 
  int     in, ik, ic,
          ih, iw,
          ir, is,
          x_x, x_y, ho, wo;

  // Quick return if possible
  if ( (n==0)||(k==0)||(c==0)||
       (h==0)||(w==0)||
       (r==0)||(s==0))
    return;

  ho = floor(((double) h + 2 * vpadding - r) / 1) + 1;
  wo = floor(((double) w + 2 * hpadding - s) / 1) + 1;

  for ( in=0;  in<n;   in++ ) 
  for ( ik=0;  ik<k;   ik++ ) 
  for ( ic=0;  ic<c;   ic++ ) 
  for ( ih=0;  ih<ho;   ih++ ) 
  for ( iw=0;  iw<wo;   iw++ ) 
  for ( ir=0;  ir<r;   ir++ ) {
    x_x = ih + ir - vpadding;
    if (0 <= x_x && x_x < h) 
      for ( is=0;  is<s;   is++ ) {
        x_y = iw + is - hpadding;
        if (0 <= x_y && x_y < w) {
           if ( tformat == NCHW )
              Ygrow_nchw(in,ik,ih,iw) += Drow_nchw(in,ic,x_x,x_y) * Frow_nchw(ik,ic,ir,is);
           else
              Ygrow_nhwc(in,ik,ih,iw) += Drow_nhwc(in,ic,x_x,x_y) * Frow_nhwc(ik,ic,ir,is);
        }
      }
  }
}  