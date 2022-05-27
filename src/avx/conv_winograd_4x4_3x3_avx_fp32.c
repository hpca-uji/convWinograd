/**
 * This file is part of convwinograd
 *
 * An implementation of the Winograd-based convolution transform
 *
 * Copyright (C) 2021-22 Universitat Politècnica de València and
 *                       Universitat Jaume I
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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include "../cblas_headers.h"

#ifdef C_AVX_FOUND

#include <xmmintrin.h>
#include <immintrin.h>
#include "avx2_transpose.h"

#else
#warning No AVX support - will not compile
#endif

extern double dclock();

#define min(a, b)     ( (a) > (b) ? (b) : (a) )
#define max(a, b)     ( (a) > (b) ? (a) : (b) )

#define Urow(a1, a2, a3, a4)  U[ (a1)*(ldU1)+(a2)*(ldU2)+(a3)*(ldU3)+(a4) ]
#define Vrow(a1, a2, a3, a4)  V[ (a1)*(ldV1)+(a2)*(ldV2)+(a3)*(ldV3)+(a4) ]
#define Mrow(a1, a2, a3, a4)  M[ (a1)*(ldM1)+(a2)*(ldM2)+(a3)*(ldM3)+(a4) ]

#ifdef TENSOR_FORMAT_NHWC
#define Drow(a1,a2,a3,a4)  D[ (a1)*(ldD1)+(a3)*(ldD2)+(a4)*(ldD3)+(a2) ]
#define Frow(a1,a2,a3,a4)  F[ (a2)*(ldF1)+(a3)*(ldF2)+(a4)*(ldF3)+(a1) ]
#define Yrow(a1,a2,a3,a4)  Y[ (a1)*(ldY1)+(a3)*(ldY2)+(a4)*(ldY3)+(a2) ]
#define drow(a1,a2)        (( fh <= a1 && a1 < oh && fw <= a2 && a2 < ow ) ? Drow(in, ic, hh + a1 - fh, ww + a2 - fw) : 0.0)
#define Mprow(a1,a2)       Mrow(a2, a1, ik, in * tile_h * tile_w + ih * tile_w + iw)
#else
#define Drow(a1, a2, a3, a4)  D[ (a1)*(ldD1)+(a2)*(ldD2)+(a3)*(ldD3)+(a4) ]
#define Frow(a1, a2, a3, a4)  F[ (a1)*(ldF1)+(a2)*(ldF2)+(a3)*(ldF3)+(a4) ]
#define Yrow(a1, a2, a3, a4)  Y[ (a1)*(ldY1)+(a2)*(ldY2)+(a3)*(ldY3)+(a4) ]
#define drow(a1, a2)        (( fh <= a1 && a1 < oh && fw <= a2 && a2 < ow ) ? Drow(in, ic, hh + a1 - fh, ww + a2 - fw) : 0.0)
#define Mprow(a1, a2)       Mrow(a2, a1, ik, in * tile_h * tile_w + ih * tile_w + iw)
#endif

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_4x4_3x3_avx_fp32_nhwc_pre
#else
void conv_winograd_4x4_3x3_avx_fp32_nchw_pre
#endif
        (int m, int r, int n, int k, int c, int kh, int kw,
         float *F, int ldF1, int ldF2, int ldF3, float *U) {
    m = 4;
    r = 3;
    const int t = m + r - 1;    // Winograd input tile size: t x t
    const int s = m;            // Winograd sliding window stride: t - (r - 1) = m
    const int vstride = 1, hstride = 1;  // Convolution stride needs to be 1

    if ((kh != r) || (kw != r)) {
        printf("*** Error: the kernel size for this version of Winograd is wrong!");
        exit(-1);
    }

    // Quick return if possible
    if ((k == 0) || (c == 0) ||
        (kh == 0) || (kw == 0))
        return;

    int ik, ic, ldU1, ldU2, ldU3, i, j;
    __m256  UX[6], WX[8];

    ldU3 = c;
    ldU2 = k * ldU3;
    ldU1 = t * ldU2;

#ifdef DEBUG
    double      t1, t2, T1;
    t1 = dclock();
#endif

    int fptw = max(1, 8 / r);
    int ct = ceil((double) c / fptw);
 
#pragma omp parallel for collapse(2) private(ik, ic, WX, UX, i) if ((k * ct) > 1)
    for (ik = 0; ik < k; ik++)
        for (ic = 0; ic < ct; ic++) {
            // U[..., ik, ic] = (G @ F[ik, ic, ...]) @ G.T
            // Load rows of F: 3x3
            // For ARM NEON, the following solution is a bit "dirty" because F has 3 elements per row only,
            // but we load four to take advantage of vector instructions
            // This may generate a core dump if we try to access in an illegal position though.
            // The alternative is to load _F2 scalar-wise. (There can be no problem with _F0 and _F1)

            // Load 4x4 blocks of 4x4 elements of F
            int max_fw = min(c - (ic*fptw), fptw), fw;

            for (fw = 0; fw < max_fw; fw++)
                for (i = 0; i < r; i++)
                    for (j = 0; j < r; j++)
                        UX[i][fw*r + j] = Frow(ik, ic*fptw + fw, i, j);

            // We are doing extra flops here: each row has only 3 valid elements but we
            // use vector instructions that operate with 4 values each. For each row/vector register, the last entry
            // is actually garbage and, therefore, will not used in the subsequent "gemm", when accessing W
            // Wi  = G_row(i)  *  [ _F0;_F1;_F2 ] (rows of F) with
            // G = [1.0,  0.0, 0.0,
            //      0.5,  0.5, 0.5,
            //      0.5, -0.5, 0.5,
            //      0.0,  0.0, 1.0];
            WX[0] = (float) (1.0 / 4.0) * UX[0];
            WX[1] = (float) (1.0 / 6.0) * (-UX[0] - UX[1] - UX[2]);
            WX[2] = WX[1] + (float) (2.0 / 6.0) * UX[1];
            WX[3] = (float) (1.0 / 24.0) * UX[0] + (float) (1.0 / 12.0) * UX[1] + (float) (1.0 / 6.0) * UX[2];
            WX[4] = WX[3] - (float) (2.0 / 12.0) * UX[1];
            WX[5] = UX[2];

            // Transpose Wk so that
            // W0, W1, W2, W3 now contain the columns of the previous Wk
            // Note that, after the transposition, W3 contains garbage
            // and it will not be used in the subsequent operations
            _MM_TRANSPOSE8_PS(WX[0], WX[1], WX[2], WX[3], WX[4], WX[5], WX[6], WX[7]);

            // Ui  = G_row(i)  *  [ _W0,_W1,_W2 ] (rows of W/cols of W before transposition)
            for (fw = 0; fw < max_fw; fw++) {
                UX[0] = (float) (1.0 / 4.0) * WX[fw*r+0];
                UX[1] = (float) (1.0 / 6.0) * (-WX[fw*r+0] - WX[fw*r+1] - WX[fw*r+2]);
                UX[2] = UX[1] + (float) (2.0 / 6.0) * WX[fw*r+1];
                UX[3] = (float) (1.0 / 24.0) * WX[fw*r+0] + (float) (1.0 / 12.0) * WX[fw*r+1] + (float) (1.0 / 6.0) * WX[fw*r+2];
                UX[4] = UX[3] - (float) (2.0 / 12.0) * WX[fw*r+1];
                UX[5] = WX[fw*r+2];

                // Scatter result in appropriate entries of U
                for (i = 0; i < t; i++)
                    for (j = 0; j < t; j++)
                        Urow(j, i, ik, (ic*fptw) + fw) = UX[i][j];
            }
        }
/*
#pragma omp parallel for collapse(2) private(ik, ic, UX, WX, i, j) if ((k * c) > 1)
    for (ik = 0; ik < k; ik++)
        for (ic = 0; ic < c; ic++) {
            // U[..., ik, ic] = (G @ F[ik, ic, ...]) @ G.T
            // Load rows of F: 3x3
            // For ARM NEON, the following solution is a bit "dirty" because F has 3 elements per row only,
            // but we load four to take advantage of vector instructions
            // This may generate a core dump if we try to access in an illegal position though.
            // The alternative is to load F2 scalar-wise. (There can be no problem with F0 and F1)
            for (i = 0; i < r; i++)
                for (j = 0; j < r; j++)
                    UX[i][j] = Frow(ik, ic, i, j);

            // We are doing extra flops here: each row has only 3 valid elements but we
            // use vector instructions that operate with 4 values each. For each row/vector register, the last entry
            // is actually garbage and, therefore, will not used in the subsequent "gemm", when accessing W
            // Wi  = G_row(i)  *  [ F0;F1;F2 ] (rows of F) with

            //   [  1./4.,      0,      0 ]     [ F01,  F01,  F02]     [W00,  W01,  W02]
            //   [ -1./6., -1./6., -1./6. ]  *  [ F10,  F11,  F12]  =  [W10,  W11,  W12]
            //   [ -1./6.,  1./6., -1./6. ]     [ F20,  F21,  F22]     [W20,  W21,  W22]
            //   [ 1./24., 1./12.,  1./6. ]                            [W30,  W31,  W32]
            //   [ 1./24.,-1./12.,  1./6. ]                            [W40,  W41,  W42]
            //   [      0,      0,      1 ]                            [W50,  W51,  W52]
            WX[0] = (float) (1.0 / 4.0) * UX[0];
            WX[1] = (float) (1.0 / 6.0) * (-UX[0] - UX[1] - UX[2]);
            WX[2] = WX[1] + (float) (2.0 / 6.0) * UX[1];
            WX[3] = (float) (1.0 / 24.0) * UX[0] + (float) (1.0 / 12.0) * UX[1] + (float) (1.0 / 6.0) * UX[2];
            WX[4] = WX[3] - (float) (2.0 / 12.0) * UX[1];
            WX[5] = UX[2];

            // Transpose Wk so that
            // W0, W1, W2, W3 now contain the columns of the previous Wk
            // Note that, after the transposition, W3 contains garbage
            // and it will not be used in the subsequent operations
            // The transposition of W4 and W5 are not performed as the register only would have 2 valid entries
            //   [W00,  W01,  W02]      [W00,  W10,  W20,  W30 | W40,  W50]
            //   [W10,  W11,  W12]  =>  [W01,  W11,  W21,  W31 | W41,  W51]
            //   [W20,  W21,  W22]      [W02,  W12,  W22,  W32 | W42,  W52]
            //   [W30,  W31,  W32]
            //   [W40,  W41,  W42]
            //   [W50,  W51,  W52]
            _MM_TRANSPOSE8_PS(WX[0], WX[1], WX[2], WX[3], WX[4], WX[5], WX[6], WX[7]);

            //  [  1./4.,      0,      0 ]     [ W00,  W10,  W20,  W30 | W40,  W50]     [ U00,  U01,  U02,  U03 | U04,  U05 ]
            //  [ -1./6., -1./6., -1./6. ]  *  [ W01,  W11,  W21,  W31 | W41,  W51]  =  [ U10,  U11,  U12,  U13 | U14,  U15 ]
            //  [ -1./6.,  1./6., -1./6. ]     [ W02,  W12,  W22,  W32 | W42,  W52]     [ U20,  U21,  U22,  U23 | U24,  U25 ]
            //  [ 1./24., 1./12.,  1./6. ]                                              [ U30,  U31,  U32,  U33 | U34,  U35 ]
            //  [ 1./24.,-1./12.,  1./6. ]                                              [ U40,  U41,  U42,  U43 | U44,  U45 ]
            //  [      0,      0,      1 ]                                              [ U50,  U51,  U52,  U53 | U54,  U55 ]
            //  Ui  = G_row(i)  *  [ W0,W1,W2 ] (rows of W/cols of W before transposition)
            UX[0] = (float) (1.0 / 4.0) * WX[0];
            UX[1] = (float) (1.0 / 6.0) * (-WX[0] - WX[1] - WX[2]);
            UX[2] = UX[1] + (float) (2.0 / 6.0) * WX[1];
            UX[3] = (float) (1.0 / 24.0) * WX[0] + (float) (1.0 / 12.0) * WX[1] + (float) (1.0 / 6.0) * WX[2];
            UX[4] = UX[3] - (float) (2.0 / 12.0) * WX[1];
            UX[5] = WX[2];

            // Scatter result in appropriate entries of U
            for (i = 0; i < t; i++)
                for (j = 0; j < t; j++)
                    Urow(i, j, ik, ic) = UX[j][i];
        }
*/
#ifdef DEBUG
    t2 = dclock();
    T1 = t2 - t1;
    float tot = T1;
    printf("%12.8f\n", T1/tot*100);
#endif
}

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_4x4_3x3_avx_fp32_nhwc_kernel
#else
void conv_winograd_4x4_3x3_avx_fp32_nchw_kernel
#endif
        (int m, int r, int n, int k, int c,
         int hi, int wi, int kh, int kw,
         int vpadding, int hpadding,
         float *D, int ldD1, int ldD2, int ldD3,
         float *Y, int ldY1, int ldY2, int ldY3,
         float *biases, float *U, float *V, float *M,
         const char relu, const char bn,
         float *running_mean, float *inv_std,
         float *gamma, float *beta) {
    m = 4;
    r = 3;
    const int t = m + r - 1;    // Winograd input tile size: t x t
    const int s = m;            // Winograd sliding window stride: t - (r - 1) = m
    const int vstride = 1, hstride = 1;  // Convolution stride needs to be 1

    if ((kh != r) || (kw != r)) {
        printf("*** Error: the kernel size for this version of Winograd is wrong!");
        exit(-1);
    }

    // Quick return if possible
    if ((n == 0) || (k == 0) || (c == 0) ||
        (hi == 0) || (wi == 0) ||
        (kh == 0) || (kw == 0))
        return;

    int tile_h, tile_w, ik, ic, in, ih, iw, hh, ww, hh_, ww_, fh, fw, oh, ow,
            ldU1, ldU2, ldU3,
            ldV1, ldV2, ldV3,
            ldM1, ldM2, ldM3,
            i, j, ho, wo, e, v,
            imtile_h, imtile_w, imt_h, imt_w, imt_hs, imt_vs, timt_h, timt_w,
            omtile_h, omtile_w, omt_h, omt_w, tomt_h, tomt_w;
    __m256  UX[6], WX[10],
            zeros = _mm256_setzero_ps();
          
#ifdef DEBUG
    double      t1, t2, T1, T2, T3, T4;
#endif

    ho = (hi + 2 * vpadding - kh) / vstride + 1;
    wo = (wi + 2 * hpadding - kw) / hstride + 1;

    tile_h = ceil(((double) hi + 2 * vpadding - t) / s) + 1;
    tile_w = ceil(((double) wi + 2 * hpadding - t) / s) + 1;

    timt_h= 1;                     timt_w= 2;                     // Number of tiles per input macrotile: height and width
    imt_h = t + (timt_h - 1) * s;  imt_w = t + (timt_w - 1) * s;  // Input macrotile height and width
    imt_vs= timt_h * s;            imt_hs= timt_w * s;            // Input Macrotile vstride and hstride
    imtile_h = ceil(((double) hi + 2 * vpadding - imt_h) / imt_vs) + 1;
    imtile_w = ceil(((double) wi + 2 * hpadding - imt_w) / imt_hs) + 1;
    
    ldU3 = c;
    ldU2 = k * ldU3;
    ldU1 = t * ldU2;

    ldV3 = (n * tile_h * tile_w);
    ldV2 = c * ldV3;
    ldV1 = t * ldV2;

    ldM3 = (n * tile_h * tile_w);
    ldM2 = k * ldM3;
    ldM1 = t * ldM2;

#ifdef DEBUG
    t1 = dclock();
#endif

#pragma omp parallel for collapse(2) private(ic, ih, hh_, hh, fh, oh, iw, ww_, ww, fw, ow, UX, WX, i, j) if ((n * c) > 1)
    for (in = 0; in < n; in++)
        for (ic = 0; ic < c; ic++)
            for (ih = 0; ih < imtile_h; ih++) {
                hh_ = min(hi, ih * imt_vs - vpadding);
                hh = max(hh_, 0);
                fh = min(max(-hh_, 0), imt_h);
                oh = max(min(hi - hh, imt_h), 0);
                oh = oh < imt_h ? oh + fh : oh;

                for (iw = 0; iw < imtile_w; iw++) {
                    ww_ = min(wi, iw * imt_hs - hpadding);
                    ww = max(ww_, 0);
                    fw = min(max(-ww_, 0), imt_w);
                    ow = max(min(wi - ww, imt_w), 0);
                    ow = ow < imt_w ? ow + fw : ow;

                    for (i = 0; i < imt_h; i++)
                        UX[i] = _mm256_setzero_ps();

                    for (i = fh; i < oh; i++)
                        for (j = fw; j < min(ow, 8); j++)
                            UX[i][j] = Drow(in, ic, hh + i - fh, ww + j - fw);

                    // WX  = Bt_row(i)  *  [ UX0;UX1;UX2;UX3;UX4;UX5 ] (rows of d), with
                    // Bt = [    4,    0,   -5,    0,    1,    0 ]
                    //      [    0,   -4,   -4,    1,    1,    0 ]
                    //      [    0,    4,   -4,   -1,    1,    0 ]
                    //      [    0,   -2,   -1,    2,    1,    0 ]
                    //      [    0,    2,   -1,   -2,    1,    0 ]
                    //      [    0,    4,    0,   -5,    0,    1 ]

                    WX[0] = 4.0 * UX[0] - 5.0 * UX[2] + UX[4];
                    WX[1] = 4.0 * (-UX[1] - UX[2]) + UX[3] + UX[4];
                    WX[2] = 4.0 * (UX[1] - UX[2]) - UX[3] + UX[4];
                    WX[3] = 2.0 * (-UX[1] + UX[3]) - UX[2] + UX[4];
                    WX[4] = 2.0 * (UX[1] - UX[3]) - UX[2] + UX[4];
                    WX[5] = 4.0 * UX[1] - 5.0 * UX[3] + UX[5];

                    for (i = min(ow, 8); i < ow; i++) {
                        WX[i][0] = 4.0 * drow(0, i) - 5.0 * drow(2, i) + drow(4, i);
                        WX[i][1] = 4.0 * (-drow(1, i) - drow(2, i)) + drow(3, i) + drow(4, i);
                        WX[i][2] = 4.0 * (drow(1, i) - drow(2, i)) - drow(3, i) + drow(4, i);
                        WX[i][3] = 2.0 * (-drow(1, i) + drow(3, i)) - drow(2, i) + drow(4, i);
                        WX[i][4] = 2.0 * (drow(1, i) - drow(3, i)) - drow(2, i) + drow(4, i);
                        WX[i][5] = 4.0 * drow(1, i) - 5.0 * drow(3, i) + drow(5, i);
                    }

                    // Transpose Wk so that
                    // W0, W1, W2, W3, W4, W5, W6, W7 now contain the columns of the previous Wk
                    _MM_TRANSPOSE8_PS(WX[0], WX[1], WX[2], WX[3], WX[4], WX[5], WX[6], WX[7]);

                    int max_mtw = min(tile_w - (iw*timt_w), timt_w), mtw;

                    // UXt  = Bt_row(i)  *  [ WX0;WX1;WX2;WX3;WX4;WX5;WX6;WX7 ] (rows of WX), with
                    // Bt = [1.0,  0.0, -1.0,  0.0,
                    //       0.0,  1.0,  1.0,  0.0,
                    //       0.0, -1.0,  1.0,  0.0,
                    //       0.0,  1.0,  0.0, -1.0];

                    for (mtw = 0; mtw < max_mtw; mtw++) { 
                        UX[0] = 4.0 * WX[mtw*m+0] - 5.0 * WX[mtw*m+2]  + WX[mtw*m+4];
                        UX[1] = 4.0 * (-WX[mtw*m+1] - WX[mtw*m+2]) + WX[mtw*m+3] + WX[mtw*m+4];
                        UX[2] = 4.0 * (WX[mtw*m+1] - WX[mtw*m+2]) - WX[mtw*m+3] + WX[mtw*m+4];
                        UX[3] = 2.0 * (-WX[mtw*m+1] + WX[mtw*m+3]) - WX[mtw*m+2] + WX[mtw*m+4];
                        UX[4] = 2.0 * (WX[mtw*m+1] - WX[mtw*m+3]) - WX[mtw*m+2] + WX[mtw*m+4];
                        UX[5] = 4.0 * WX[mtw*m+1] - 5.0 * WX[mtw*m+3] + WX[mtw*m+5];

                        int ix = in * tile_h * tile_w + (iw * timt_w + mtw);
                        for (i = 0; i < t; i++)
                            for (j = 0; j < t; j++)
                                Vrow(i, j, ic, ix + ih * timt_h * tile_w) = UX[j][i];
                    }
                }
            }
#ifdef DEBUG
    t2 = dclock();
    T2 = t2 - t1;
    t1 = dclock();
#endif

/*
  #define    GRP_COUNT    1

  int    m_[] = {k};
  int    n_[] = {(n * tile_h * tile_w)};
  int    k_[] = {c};

  int    lda_[] = {c};
  int    ldb_[] = {(n * tile_h * tile_w)};
  int    ldc_[] = {(n * tile_h * tile_w)};

  CBLAS_TRANSPOSE    transA[] = {CblasNoTrans};
  CBLAS_TRANSPOSE    transB[] = {CblasNoTrans};

  float    alpha[] = {1.0};
  float    beta_[] = {0.0};

  int    size_per_group[] = {t * t};

  float    *a_array_[t*t], *b_array_[t*t], *c_array_[t*t];
  for (e = 0; e < t; e++) {
    for (v = 0; v < t; v++) {
       a_array_[e * t + v] = &Urow(e, v, 0, 0);
       b_array_[e * t + v] = &Vrow(e, v, 0, 0);
       c_array_[e * t + v] = &Mrow(e, v, 0, 0);
    }
  }
  const float **a_array=(const float **)a_array_, **b_array=(const float **)b_array_;

  // Call cblas_dgemm_batch
  cblas_sgemm_batch ( CblasRowMajor, transA, transB,
          m_, n_, k_,
          alpha, a_array, lda_,
                 b_array, ldb_,
          beta_, c_array_, ldc_,
          GRP_COUNT,
          size_per_group);
*/

#pragma omp parallel for collapse(2) private(e, v)
    for (e = 0; e < t; e++)
        for (v = 0; v < t; v++) {
            // M[e, v] = U[e, v] @ V[e, v]
            // Store M so that the computation in the block of nested loops after the following computation is contiguous
            // This is different from Manel's implementation in Python and it means we are actually computing
            //     M[..., e, v] = U[e, v] @ V[e, v]
#if CBLAS_TYPE_CBLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  k, (n * tile_h * tile_w), c,
                  1.0, &Urow(e, v, 0, 0), c,
                       &Vrow(e, v, 0, 0), (n * tile_h * tile_w),
                  0.0, &Mrow(e, v, 0, 0), (n * tile_h * tile_w) );
#else
            gemm('R', 'R', 'R',
                 'N', 'N',
                 k, (n * tile_h * tile_w), c,
                 1.0, &Urow(e, v, 0, 0), c,
                 &Vrow(e, v, 0, 0), (n * tile_h * tile_w),
                 0.0, &Mrow(e, v, 0, 0), (n * tile_h * tile_w));
#endif
        }
#ifdef DEBUG
    t2 = dclock();
    T3 = t2 - t1;
    t1 = dclock();
#endif

#pragma omp parallel for collapse(2) private(in, ik, ih, iw, UX, WX, hh, ww, i, j) if ((n * k) > 1)
    for (in = 0; in < n; in++)
        for (ik = 0; ik < k; ik++)
            for (ih = 0; ih < tile_h; ih++)
                for (iw = 0; iw < tile_w; iw++) {
                    // Z = (At @ M[..., ik, in * tile_h * tile_w + ih * tile_w + iw]) @ At.T
                    // Take advantage that because of the change in the previous block of nested loops, M is now contiguous in memory.
                    // Therefore, we are actually computing the following:
                    //     Z = (At @ M[in * tile_h * tile_w + ih * tile_w + iw, ik, ...]) @ At.T

                    // Load rows of M: 6x6
                    for (i = 0; i < t; i++)
                        for (j = 0; j < t; j++)
                            UX[i][j] = Mrow(j, i, ik, in * tile_h * tile_w + ih * tile_w + iw);

                    // W_i  = A_row(i)  *  [ M0;M1;M2;M3 ] (rows of M), with
                    //    [      1,      1,      1,      1,      1,      0 ]     [ M00,  M01,  M02,  M03 | M04,  M05 ]     [ W00,  W01,  W02,  W03 | W04,  W05 ]
                    //    [      0,      1,     -1,      2,     -2,      0 ]  *  [ M10,  M11,  M12,  M13 | M14,  M15 ]  =  [ W10,  W11,  W12,  W13 | W14,  W15 ]
                    //    [      0,      1,      1,      4,      4,      0 ]     [ M20,  M21,  M22,  M23 | M24,  M25 ]     [ W20,  W21,  W22,  W23 | W24,  W25 ]
                    //    [      0,      1,     -1,      8,     -8,      1 ]     [ M30,  M31,  M32,  M33 | M34,  M35 ]     [ W30,  W31,  W32,  W33 | W34,  W35 ]
                    //                                                           [ M40,  M41,  M42,  M43 | M44,  M45 ]
                    //                                                           [ M50,  M51,  M52,  M53 | M54,  M55 ]
                    WX[0] = UX[0] + UX[1] + UX[2] + UX[3] + UX[4];
                    WX[1] = UX[1] - UX[2] + 2.0 * (UX[3] - UX[4]);
                    WX[2] = UX[1] + UX[2] + 4.0 * (UX[3] + UX[4]);
                    WX[3] = UX[1] - UX[2] + 8.0 * (UX[3] - UX[4]) + UX[5];

                    // Transpose Wk so that
                    // W0, W1, W2, W3 now contain the columns of the previous Wk
                    _MM_TRANSPOSE8_PS(WX[0], WX[1], WX[2], WX[3], WX[4], WX[5], WX[6], WX[7]);

                    // In contrast with cases 1) and 2), in this case we do not use vector instructions for this second gemm as
                    // the result is only 2x2 and we would be doing many innecessary flops
                    //    [      1,      1,      1,      1,      1,      0 ]     [ W00,  W10,  W02,  W03 ]     [ Z00,  Z01,  Z02,  Z03 ]
                    //    [      0,      1,     -1,      2,     -2,      0 ]  *  [ W01,  W11,  W12,  W13 ]  =  [ Z10,  Z11,  Z12,  Z13 ]
                    //    [      0,      1,      1,      4,      4,      0 ]     [ W02,  W12,  W22,  W23 ]     [ Z20,  Z21,  Z22,  Z23 ]
                    //    [      0,      1,     -1,      8,     -8,      1 ]     [ W03,  W13,  W32,  W33 ]     [ Z30,  Z31,  Z32,  Z33 ]
                    //                                                           [ W04,  W14,  W42,  W43 ]
                    //                                                           [ W05,  W15,  W52,  W53 ]
                    UX[0] = WX[0] + WX[1] + WX[2] + WX[3] + WX[4];
                    UX[1] = WX[1] - WX[2] + 2.0 * (WX[3] - WX[4]);
                    UX[2] = WX[1] + WX[2] + 4.0 * (WX[3] + WX[4]);
                    UX[3] = WX[1] - WX[2] + 8.0 * (WX[3] - WX[4]) + WX[5];

                    if (biases != NULL)
                        for (i = 0; i < m; i++)
                            UX[i] = UX[i] + biases[ik];

                    if (bn == 'T')
                        for (i = 0; i < m; i++)
                            UX[i] = (((UX[i] - running_mean[ik]) * inv_std[ik]) * gamma[ik]) + beta[ik];

                    if (relu == 'T')
                        for (i = 0; i < m; i++)
                            UX[i] = _mm256_max_ps(UX[i], zeros);

                    hh = ih * s;
                    ww = iw * s;
                    // Yw[n, k, hh:hh+m, ww:ww+m] = Z[:min(m, H-hh), :min(m, W-ww)]
                    for (i = 0; i < min(m, ho - hh); i++)
                        for (j = 0; j < min(m, wo - ww); j++)
                            Yrow(in, ik, hh + i, ww + j) = UX[i][j];
                }
#ifdef DEBUG
    t2 = dclock();
    T4 = t2 - t1;
    float tot = T2 + T3 + T4;
    printf("%12.8f %12.8f %12.8f\n", T2/tot*100, T3/tot*100, T4/tot*100);
#endif
}

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_4x4_3x3_avx_fp32_nhwc
#else
void conv_winograd_4x4_3x3_avx_fp32_nchw
#endif
        (int m, int r, int n, int k, int c,
         int hi, int wi, int kh, int kw,
         int vpadding, int hpadding,
         float *D, int ldD1, int ldD2, int ldD3,
         float *F, int ldF1, int ldF2, int ldF3,
         float *Y, int ldY1, int ldY2, int ldY3,
         float *biases, float *Bt, float *G, float *At,
         float *U, float *V, float *M,
         const char relu, const char bn,
         float *running_mean, float *inv_std,
         float *gamma, float *beta) {

#ifdef TENSOR_FORMAT_NHWC
    conv_winograd_4x4_3x3_avx_fp32_nhwc_pre
#else
    conv_winograd_4x4_3x3_avx_fp32_nchw_pre
#endif
        (m, r, n, k, c, kh, kw, F, ldF1, ldF2, ldF3, U);

#ifdef TENSOR_FORMAT_NHWC
    conv_winograd_4x4_3x3_avx_fp32_nhwc_kernel
#else
    conv_winograd_4x4_3x3_avx_fp32_nchw_kernel
#endif
        (m, r, n, k, c, hi, wi, kh, kw, vpadding, hpadding,
         D, ldD1, ldD2, ldD3, Y, ldY1, ldY2, ldY3,
         biases, U, V, M, relu, bn, running_mean, inv_std,
         gamma, beta);
}
