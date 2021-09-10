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
#include <math.h>
#include <string.h>

// #include "dtypes.h"
#if defined(EXTERN_CBLAS)
#include <cblas.h>
#elif !defined(ARM_NEON)
#include "gemm.h"
#endif

#include <riscv_vector.h>

void transpose_4x4_vfloat32m1_t(vfloat32m1_t * R0, vfloat32m1_t * R1,
                                vfloat32m1_t * R2, vfloat32m1_t * R3){
     vfloat32m1_t C0, C1, C2, C3;
     C0 = vrgather_vx_f32m1(*R3, 0, 4);
     C0 = vrgather_vx_f32m1(*R2, 0, 3);
     C0 = vrgather_vx_f32m1(*R1, 0, 2);
     C0 = vrgather_vx_f32m1(*R0, 0, 1);
     C1 = vrgather_vx_f32m1(*R3, 1, 4);
     C1 = vrgather_vx_f32m1(*R2, 1, 3);
     C1 = vrgather_vx_f32m1(*R1, 1, 2);
     C1 = vrgather_vx_f32m1(*R0, 1, 1);
     C2 = vrgather_vx_f32m1(*R3, 2, 4);
     C2 = vrgather_vx_f32m1(*R2, 2, 3);
     C2 = vrgather_vx_f32m1(*R1, 2, 2);
     C2 = vrgather_vx_f32m1(*R0, 2, 1);
     C3 = vrgather_vx_f32m1(*R3, 3, 4);
     C3 = vrgather_vx_f32m1(*R2, 3, 3);
     C3 = vrgather_vx_f32m1(*R1, 3, 2);
     C3 = vrgather_vx_f32m1(*R0, 3, 1);

     *R0 = C0;
     *R1 = C1;
     *R2 = C2;
     *R3 = C3;
}

#define dabs(a)      ( (a) > 0.0 ? (a) :-(a) )
#define min(a,b)     ( (a) > (b) ? (b) : (a) )
#define max(a,b)     ( (a) > (b) ? (a) : (b) )

#define Drow(a1,a2,a3,a4)  D[ (a1)*(ldD1)+(a2)*(ldD2)+(a3)*(ldD3)+(a4) ]
#define Frow(a1,a2,a3,a4)  F[ (a1)*(ldF1)+(a2)*(ldF2)+(a3)*(ldF3)+(a4) ]
#define Yrow(a1,a2,a3,a4)  Y[ (a1)*(ldY1)+(a2)*(ldY2)+(a3)*(ldY3)+(a4) ]
#define Urow(a1,a2,a3,a4)  U[ (a1)*(ldU1)+(a2)*(ldU2)+(a3)*(ldU3)+(a4) ]
#define Vrow(a1,a2,a3,a4)  V[ (a1)*(ldV1)+(a2)*(ldV2)+(a3)*(ldV3)+(a4) ]
#define Mrow(a1,a2,a3,a4)  M[ (a1)*(ldM1)+(a2)*(ldM2)+(a3)*(ldM3)+(a4) ]

#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]

void conv_winograd_2x2_3x3_nchw_neon_fp32(int m, int r, int n, int k, int c,
                   int hi, int wi, int kh, int kw,
                   int vpadding, int hpadding,
                   float *D, int ldD1, int ldD2, int ldD3,
                   float *F, int ldF1, int ldF2, int ldF3,
                   float *Y, int ldY1, int ldY2, int ldY3,
                   float *biases, float *Bt, float *G, float *At,
                   float *U,  float *V, float *M, float *MA2,
                   const char relu, const char bn,
                   float *running_mean, float *inv_std, 
                   float *gamma, float *beta)
{
  m = 2; r = 3;
  const   int t = m + r - 1;    // Winograd input tile size: t x t
  const   int s = m;            // Winograd sliding window stride: t - (r - 1) = m
  const   int vstride = 1, hstride = 1;  // Convolution stride needs to be 1
 
  if ((kh != r)||(kw != r)) {
    printf("*** Error: the kernel size for this version of Winograd is wrong!");
    exit(-1);
  }

  // Quick return if possible
 if ( (n==0)||(k==0)||(c==0)||
       (hi==0)||(wi==0)||
       (kh==0)||(kw==0) )
    return;

  int          tile_h, tile_w, ik, ic, in, ih, iw, hh, ww, hh_, ww_, fh, fw, oh, ow,
               ldU1, ldU2, ldU3,
               ldV1, ldV2, ldV3,
               ldM1, ldM2, ldM3,
               i, j, ho, wo, e, v,
               vlength = 4;
  float        d[t*t], *Fptr, *dptr, *Mptr,
               init[4] = {0.0,0.0,0.0,0.0},
               *init_ptr = init,
               W00, W01, W02, W03,
               W10, W11, W12, W13,
               Z_tmp[4];
  size_t       vl =  vsetvl_e32m1(vlength);
  vfloat32m1_t F0, F1, F2,
               d0, d1, d2, d3,
               U0, U1, U2, U3,
               M0, M1, M2, M3,
               W0, W1, W2, W3, 
               Z,
               zeros = vle32_v_f32m1(init_ptr, vl);

  ho = floor(((double) hi + 2 * vpadding - kh) / vstride) + 1;
  wo = floor(((double) wi + 2 * hpadding - kw) / hstride) + 1;

  tile_h = ceil(((double) hi + 2 * vpadding - t) / s) + 1;
  tile_w = ceil(((double) wi + 2 * hpadding - t) / s) + 1;

  ldU3 = c;
  ldU2 = k*ldU3;
  ldU1 = t*ldU2;

  ldV3 = (n * tile_h * tile_w);
  ldV2 = c*ldV3;
  ldV1 = t*ldV2;

  ldM3 = t;
  ldM2 = t*ldM3;
  ldM1 = k*ldM2;

  // This is not necessary as all entries of Y are written after computing the Winograd algorithm
  // for (ik = 0; ik < k; ik++)
  //  for (in = 0; in < k; in++)
  //    for (ih = 0; ih < ho; ih++)
  //      for (iw = 0; iw < wo; iw++)
  //        Yrow(in, ik, ih, iw) = 0.0;
 
  for (ik = 0; ik < k; ik++)
    for (ic = 0; ic < c; ic++) {
      // U[..., ik, ic] = (G @ F[ik, ic, ...]) @ G.T

      // Load rows of F: 3x3
      // For ARM NEON, the following solution is a bit "dirty" because F has 3 elements per row only, 
      // but we load four to take advantage of vector instructions
      // This may generate a core dump if we try to access in an illegal position though.
      // The alternative is to load F2 scalar-wise. (There can be no problem with F0 and F1)
      Fptr = &Frow(ik,ic,0,0);
      F0 = vle32_v_f32m1(&Fptr[0], 3);
      F1 = vle32_v_f32m1(&Fptr[3], 3);
      F2 = vle32_v_f32m1(&Fptr[6], 3);

      // We are doing extra flops here: each row has only 3 valid elements but we
      // use vector instructions that operate with 4 values each. For each row/vector register, the last entry
      // is actually garbage and, therefore, will not used in the subsequent "gemm", when accessing W
      // Wi  = G_row(i)  *  [ F0;F1;F2 ] (rows of F) with
      // G = [1.0,  0.0, 0.0,
      //      0.5,  0.5, 0.5,
      //      0.5, -0.5, 0.5,
      //      0.0,  0.0, 1.0];
      W0 = F0;
      W1 = 0.5 * (F0 + F1 + F2);
      W2 = 0.5 * (F0 - F1 + F2);
      W3 = F2;

      // Transpose Wk so that
      // W0, W1, W2, W3 now contain the columns of the previous Wk
      // Note that, after the transposition, W3 contains garbage
      // and it will not be used in the subsequent operations
      transpose_4x4_vfloat32m1_t(&W0,  &W1, &W2, &W3 );

      // Ui  = G_row(i)  *  [ W0,W1,W2 ] (rows of W/cols of W before transposition)
      U0 = W0;
      U1 = 0.5 * (W0 + W1 + W2);
      U2 = 0.5 * (W0 - W1 + W2);
      U3 = W2;

      // Scatter result in appropriate entries of U
      for (i = 0; i < 4; i++) {
        vsse32_v_f32m1(&Urow(i, 0, ik, ic), 0, U0, i+1);
        vsse32_v_f32m1(&Urow(i, 1, ik, ic), 0, U1, i+1);
        vsse32_v_f32m1(&Urow(i, 2, ik, ic), 0, U2, i+1);
        vsse32_v_f32m1(&Urow(i, 3, ik, ic), 0, U3, i+1);
      }
    }

  for (in = 0; in < n; in++)
    for (ic = 0; ic < c; ic++)
      for (ih = 0; ih < tile_h; ih++) {
        hh_= min(hi, ih * s - vpadding);
        hh = max(hh_, 0);
        fh = min(max(-hh_, 0), t);
        oh = max(min(hi - hh, t) - fh, 0);

        for (iw = 0; iw < tile_w; iw++) {
          ww_= min(wi, iw * s - hpadding);
          ww = max(ww_, 0);
          fw = min(max(-ww_, 0), t);
          ow = max(min(wi - ww, t) - fw, 0);

          for (i = 0; i < oh; i++)
            for (j = 0; j < ow; j++)
              d[(fh + i) * t + (fw + j)] = Drow(in, ic, hh + i, ww + j);

          //   0  0  0
          //   X  X  X
          //   X  X  X
          // if 0 <= fh:
          //    d[:fh, ...] = 0
          for (i = 0; i < fh; i++)
            for (j = 0; j < t; j++)
              d[i * t + j] = 0.0;

          //   0  0  0
          //   X  X  X
          //   0  0  0
          // if fh + oh < t:
          //     d[fh+oh:, ...] = 0
          for (i = fh + oh; i < t; i++)
            for (j = 0; j < t; j++)
              d[i * t + j] = 0.0;

          //   0  0  0
          //   0  X  X
          //   0  0  0
          // if 0 <= fw:
          //     d[fh:fh+oh, :fw] = 0
          for (i = fh; i < min(fh+oh, t); i++)
            for (j = 0; j < fw; j++)
              d[i * t + j] = 0.0;

          //   0  0  0
          //   0  X  0
          //   0  0  0
          // if fw + ow < t:
          //     d[fh:fh+oh, fw+ow:] = 0
          for (i = fh; i < min(fh+oh, t); i++)
            for (j = fw + ow; j < t; j++)
              d[i * t + j] = 0.0;
          
          // V[..., ic, in * tile_h * tile_w + ih * tile_w + iw] = (Bt @ d) @ Bt.T

          // Load rows of d: 4x4 (This is much more convenient than the case with 3x3 F.)
          //
          // WARNING: We should replace this vector loads with a direct constructions of d from the entries of Drow
          // and therefore avoid the duplicated memory accesses
          //
          dptr = d;
          d0   = vle32_v_f32m1(&dptr[0], vl);
          d1   = vle32_v_f32m1(&dptr[4], vl);
          d2   = vle32_v_f32m1(&dptr[8], vl);
          d3   = vle32_v_f32m1(&dptr[12], vl);

          // Wi  = Bt_row(i)  *  [ d0;d1;d2;d3 ] (rows of d), with
          // Bt = [1.0,  0.0, -1.0,  0.0,
          //       0.0,  1.0,  1.0,  0.0,
          //       0.0, -1.0,  1.0,  0.0,
          //       0.0,  1.0,  0.0, -1.0];
          W0 =  d0      - d2;
          W1 =       d1 + d2;
          W2 =     - d1 + d2;
          W3 =       d1      - d3;

          // Transpose Wk so that
          // W0, W1, W2, W3 now contain the columns of the previous Wk
          transpose_4x4_vfloat32m1_t(&W0,  &W1, &W2, &W3 );
          
          // U_i  = Bt_row(i)  *  [ W0,W1,W2,W3 ] (rows of W/cols of W before transposition)
          U0 =  W0      - W2;
          U1 =       W1 + W2;
          U2 =     - W1 + W2;
          U3 =       W1      - W3;

          // Scatter result in appropriate entries of V
          for (i = 0; i < 4; i++) {
            vsse32_v_f32m1(&Vrow(i, 0, ic, in * tile_h * tile_w + ih * tile_w + iw), 0, U0, i+1);
            vsse32_v_f32m1(&Vrow(i, 1, ic, in * tile_h * tile_w + ih * tile_w + iw), 0, U1, i+1);
            vsse32_v_f32m1(&Vrow(i, 2, ic, in * tile_h * tile_w + ih * tile_w + iw), 0, U2, i+1);
            vsse32_v_f32m1(&Vrow(i, 3, ic, in * tile_h * tile_w + ih * tile_w + iw), 0, U3, i+1);
          }
        }
     }

  for (e = 0; e < t; e++)
    for (v = 0; v < t; v++) {
      // M[e, v] = U[e, v] @ V[e, v]
      // Store M so that the computation in the block of nested loops after the following computation is contiguous
      // This is different from Manel's implementation in Python and it means we are actually computing
      //     M[..., e, v] = U[e, v] @ V[e, v]
#if defined(EXTERN_CBLAS)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            k, (n * tile_h * tile_w), c,
            1.0, &Urow(e, v, 0, 0), c,
                 &Vrow(e, v, 0, 0), (n * tile_h * tile_w),
            0.0, MA2, (n * tile_h * tile_w) );
#else
      gemm( 'R', 'R', 'R',
            'N', 'N',
            k, (n * tile_h * tile_w), c,
            1.0, &Urow(e, v, 0, 0), c,
                 &Vrow(e, v, 0, 0), (n * tile_h * tile_w),
            0.0, MA2, (n * tile_h * tile_w) );
#endif
      for (i = 0; i < (n * tile_h * tile_w); i++)
        for (j = 0; j < k; j++)
           Mrow(i, j, v, e) = MA2[j * (n * tile_h * tile_w) + i];
    }

  for (in = 0; in < n; in++)
    for (ik = 0; ik < k; ik++)
      for (ih = 0; ih < tile_h; ih++)
        for (iw = 0; iw < tile_w; iw++) {
          // Z = (At @ M[..., ik, in * tile_h * tile_w + ih * tile_w + iw]) @ At.T
          // Take advantage that because of the change in the previous block of nested loops, M is now contiguous in memory.
          // Therefore, we are actually computing the following:
          //     Z = (At @ M[in * tile_h * tile_w + ih * tile_w + iw, ik, ...]) @ At.T

          // Load rows of M: 4x4
          Mptr = &Mrow(in * tile_h * tile_w + ih * tile_w + iw, ik, 0, 0);
          M0 = vle32_v_f32m1(&Mptr[0], vl);
          M1 = vle32_v_f32m1(&Mptr[4], vl);
          M2 = vle32_v_f32m1(&Mptr[8], vl);
          M3 = vle32_v_f32m1(&Mptr[12], vl);

          // W_i  = A_row(i)  *  [ M0;M1;M2;M3 ] (rows of M), with
          // At  = [1.0, 1.0,  1.0,  0.0, 
          //        0.0, 1.0, -1.0, -1.0];
          W0 = M0 + M1 + M2;
          W1 =      M1 - M2 - M3;

          // In contrast with cases 1) and 2), in this case we do not use vector instructions for this second gemm as
          // the result is only 2x2 and we would be doing many innecessary flops
          vsse32_v_f32m1(&W00, 0, W0, 1);
          vsse32_v_f32m1(&W01, 0, W0, 2);
          vsse32_v_f32m1(&W02, 0, W0, 3);
          vsse32_v_f32m1(&W03, 0, W0, 4);
          vsse32_v_f32m1(&W10, 0, W1, 1);
          vsse32_v_f32m1(&W11, 0, W1, 2);
          vsse32_v_f32m1(&W12, 0, W1, 3);
          vsse32_v_f32m1(&W13, 0, W1, 4);

          // OJO! Se intenta sumar registros vectoriales!
          //Z_tmp is a 4 float length vector
          Z_tmp[0] = W00 + W01 + W02;
          Z_tmp[1] =       W01 - W02 - W03;
          Z_tmp[2] = W10 + W11 + W12;
          Z_tmp[3] =       W11 - W12 - W13;
          Z = vle32_v_f32m1(&Z_tmp[0], vl);


          if ( biases != NULL )
            Z = Z + biases[ik];

          if ( bn == 'T' )
            Z = (((Z - running_mean[ik]) * inv_std[ik]) * gamma[ik]) + beta[ik];

          if ( relu == 'T' )
              Z = vfmax_vf_f32m1 (Z, 0, vl);

          hh = ih * s;
          ww = iw * s;
          // Yw[n, k, hh:hh+m, ww:ww+m] = Z[:min(m, H-hh), :min(m, W-ww)]
          for (i = 0; i < min(m, ho-hh); i++)
            for (j = 0; j < min(m, wo-ww); j++)
              vsse32_v_f32m1(&Yrow(in, ik, hh + i, ww + j), 0, Z, j * m + i + 1);
        }
}
