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
void conv_winograd_2x2_5x5_avx_fp32_nhwc_pre
#else
void conv_winograd_2x2_5x5_avx_fp32_nchw_pre
#endif
        (int m, int r, int n, int k, int c, int kh, int kw,
         float *F, int ldF1, int ldF2, int ldF3, float *U) {
    m = 2;
    r = 5;
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
    __m256  FX[5], WX[8], UX[6];

    ldU3 = c;
    ldU2 = k * ldU3;
    ldU1 = t * ldU2;

#pragma omp parallel for collapse(2) private(ik, ic, FX, WX, UX, i, j) if ((k * c) > 1)
    for (ik = 0; ik < k; ik++)
        for (ic = 0; ic < c; ic++) {
            // U[..., ik, ic] = (G @ F[ik, ic, ...]) @ G.T

            // Load rows of F: 3x3
            // The following solution is a bit "dirty" because F has 3 elements per row only,
            // but we load four to take advantage of vector instructions
            // This may generate a core dump if we try to access in an illegal position though.
            // The alternative is to load F2 scalar-wise. (There can be no problem with F0 and F1)
            for (i = 0; i < r; i++)
                FX[i] = _mm256_setzero_ps();

            for (i = 0; i < r; i++)
                for (j = 0; j < r; j++)
                    FX[i][j] = Frow(ik, ic, i, j);

            // We are doing extra flops here: each row has only 3 valid elements but we
            // use vector instructions that operate with 4 values each. For each row/vector register, the last entry
            // is actually garbage and, therefore, will not used in the subsequent "gemm", when accessing W
            // Wi  = G_row(i)  *  [ F0;F1;F2;F3;F4 ] (rows of F) with
            // Gt = [  1./4.,      0,      0,      0,      0   ]
            //      [ -1./6., -1./6., -1./6., -1./6., -1./6.   ]
            //      [ -1./6.,  1./6., -1./6.,  1./6., -1./6.   ]
            //      [ 1./24., 1./12.,  1./6.,  1./3.,  2./3.   ]
            //      [ 1./24.,-1./12.,  1./6., -1./3.,  2./3.   ]
            //      [      0,      0,      0,      0,      1   ]
            WX[0] = (float) (1.0 / 4.0) * FX[0];
            WX[1] = (float) (1.0 / 6.0) * (-FX[0] - FX[1] - FX[2] - FX[3] - FX[4]);
            WX[2] = WX[1] + (float) (2.0 / 6.0) * (FX[1] + FX[3]);
            WX[3] = (float) (1.0 / 24.0) * FX[0] + (float) (1.0 / 12.0) * FX[1] + (float) (1.0 / 6.0) * FX[2] +
                    (float) (1.0 / 3.0) * FX[3] + (float) (2.0 / 3.0) * FX[4];
            WX[4] = WX[3] - (float) (2.0 / 12.0) * FX[1] - (float) (2.0 / 3.0) * FX[3];
            WX[5] = FX[4];
            WX[6] = _mm256_setzero_ps();
            WX[7] = _mm256_setzero_ps();

            // Transpose Wk so that
            // W0, W1, W2, W3 now contain the columns of the previous Wk
            // Note that, after the transposition, W3 contains garbage
            // and it will not be used in the subsequent operations
            // The transposition of W4 and W5 are not performed as the register only would have 2 valid entries
            _MM_TRANSPOSE8_PS(WX[0], WX[1], WX[2], WX[3], WX[4], WX[5], WX[6], WX[7]);

            // Gt = [  1./4.,      0,      0,      0,      0   ]
            //      [ -1./6., -1./6., -1./6., -1./6., -1./6.   ]
            //      [ -1./6.,  1./6., -1./6.,  1./6., -1./6.   ]
            //      [ 1./24., 1./12.,  1./6.,  1./3.,  2./3.   ]
            //      [ 1./24.,-1./12.,  1./6., -1./3.,  2./3.   ]
            //      [      0,      0,      0,      0,      1   ]
            UX[0] = (float) (1.0 / 4.0) * WX[0];
            UX[1] = (float) (1.0 / 6.0) * (-WX[0] - WX[1] - WX[2] - WX[3] - WX[4]);
            UX[2] = UX[1] + (float) (2.0 / 6.0) * (WX[1] + WX[3]);
            UX[3] = (float) (1.0 / 24.0) * WX[0] + (float) (1.0 / 12.0) * WX[1] + (float) (1.0 / 6.0) * WX[2] +
                    (float) (1.0 / 3.0) * WX[3] + (float) (2.0 / 3.0) * WX[4];
            UX[4] = UX[3] - (float) (2.0 / 12.0) * WX[1] - (float) (2.0 / 3.0) * WX[3];
            UX[5] = WX[4];

            // Scatter result in appropriate entries of U
            for (i = 0; i < t; i++)
                for (j = 0; j < t; j++)
                    Urow(j, i, ik, ic) = UX[i][j];
        }
}

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_2x2_5x5_avx_fp32_nhwc_post
#else
void conv_winograd_2x2_5x5_avx_fp32_nchw_post
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
    m = 2;
    r = 5;
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
            imtile_h, imtile_w, imt_h, imt_w, imt_hs, imt_vs, timt_h, timt_w;
    __m128  Z, zeros = _mm_set_ps(0.0, 0.0, 0.0, 0.0);
    __m256  UX[6], WX[8], MX[6];

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
                        for (j = fw; j < ow; j++)
                            UX[i][j] = Drow(in, ic, hh + i - fh, ww + j - fw);

                    // Wi  = Bt_row(i)  *  [ d0;d1;d2;d3;d4,d5 ] (rows of d), with
                    // Bt = [    4,    0,   -5,    0,    1,    0 ]
                    //      [    0,   -4,   -4,    1,    1,    0 ]
                    //      [    0,    4,   -4,   -1,    1,    0 ]
                    //      [    0,   -2,   -1,    2,    1,    0 ]
                    //      [    0,    2,   -1,   -2,    1,    0 ]
                    //      [    0,    4,    0,   -5,    0,    1 ]
                    for (i = 0; i < timt_h; i++) {
                        WX[i*t+0] = 4.0 * UX[i*s+0] - 5.0 * UX[i*s+2] + UX[i*s+4];
                        WX[i*t+1] = 4.0 * (-UX[i*s+1] - UX[i*s+2]) + UX[i*s+3] + UX[i*s+4];
                        WX[i*t+2] = 4.0 * (UX[i*s+1] - UX[i*s+2]) - UX[i*s+3] + UX[i*s+4];
                        WX[i*t+3] = 2.0 * (-UX[i*s+1] + UX[i*s+3]) - UX[i*s+2] + UX[i*s+4];
                        WX[i*t+4] = 2.0 * (UX[i*s+1] - UX[i*s+3]) - UX[i*s+2] + UX[i*s+4];
                        WX[i*t+5] = 4.0 * UX[i*s+1] - 5.0 * UX[i*s+3] + UX[i*s+5];
/*
                        WX[i*t+6] = _mm256_setzero_ps();
                        WX[i*t+7] = _mm256_setzero_ps();
*/
                    }

                    // Transpose Wk so that
                    // W0, W1, W2, W3 now contain the columns of the previous Wk
                    _MM_TRANSPOSE8_PS(WX[0], WX[1], WX[2], WX[3], WX[4], WX[5], WX[6], WX[7]);

                    // U_i  = Bt_row(i)  *  [ W0,W1,W2,W3 ] (rows of W/cols of W before transposition)
                    // Bt = [    4,    0,   -5,    0,    1,    0 ]
                    //      [    0,   -4,   -4,    1,    1,    0 ]
                    //      [    0,    4,   -4,   -1,    1,    0 ]
                    //      [    0,   -2,   -1,    2,    1,    0 ]
                    //      [    0,    2,   -1,   -2,    1,    0 ]
                    //      [    0,    4,    0,   -5,    0,    1 ]

                    int max_mth = min(tile_h - (ih*timt_h), timt_h), mth;
                    int max_mtw = min(tile_w - (iw*timt_w), timt_w), mtw;

                    for (mtw = 0; mtw < max_mtw; mtw++) {
                        UX[0] = 4.0 * WX[mtw*s+0] - 5.0 * WX[mtw*s+2] + WX[mtw*s+4];
                        UX[1] = 4.0 * (-WX[mtw*s+1] - WX[mtw*s+2]) + WX[mtw*s+3] + WX[mtw*s+4];
                        UX[2] = 4.0 * (WX[mtw*s+1] - WX[mtw*s+2]) - WX[mtw*s+3] + WX[mtw*s+4];
                        UX[3] = 2.0 * (-WX[mtw*s+1] + WX[mtw*s+3]) - WX[mtw*s+2] + WX[mtw*s+4];
                        UX[4] = 2.0 * (WX[mtw*s+1] - WX[mtw*s+3]) - WX[mtw*s+2] + WX[mtw*s+4];
                        UX[5] = 4.0 * WX[mtw*s+1] - 5.0 * WX[mtw*s+3] + WX[mtw*s+5];

                        int ix = in * tile_h * tile_w + (iw*timt_w + mtw);
                        for (mth = 0; mth < max_mth; mth++)
                            for (i = 0; i < t; i++)
                                for (j = 0; j < t; j++)
                                    Vrow(i, j, ic, ix + (ih*timt_h + mth) * tile_w) = UX[j][mth*t + i];
                    }
                }
            }
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
#pragma omp parallel for collapse(2) private(in, ik, ih, iw, MX, WX, Z, hh, ww, i, j) if ((n * k) > 1)
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
                            MX[j][i] = Mrow(i, j, ik, in * tile_h * tile_w + ih * tile_w + iw);

                    // W_i  = A_row(i)  *  [ M0;M1;M2;M3 ] (rows of M), with
                    // At = [      1,      1,      1,      1,      1,      0 ]
                    //      [      0,      1,     -1,      2,     -2,      1 ]
                    WX[0] = MX[0] + MX[1] + MX[2] + MX[3] + MX[4];
                    WX[1] = MX[1] - MX[2] + 2.0 * (MX[3] - MX[4]) + MX[5];

                    // In contrast with cases 1) and 2), in this case we do not use vector instructions for this second gemm as
                    // the result is only 2x2 and we would be doing many innecessary flops
                    // At = [      1,      1,      1,      1,      1,      0 ]
                    //      [      0,      1,     -1,      2,     -2,      1 ]
                    Z[0] = WX[0][0] + WX[0][1] + WX[0][2] + WX[0][3] + WX[0][4];
                    Z[2] = WX[1][0] + WX[1][1] + WX[1][2] + WX[1][3] + WX[1][4];

                    Z[1] = WX[0][1] - WX[0][2] + 2.0 * WX[0][3] - 2.0 * WX[0][4] + WX[0][5];
                    Z[3] = WX[1][1] - WX[1][2] + 2.0 * WX[1][3] - 2.0 * WX[1][4] + WX[1][5];

                    if (biases != NULL)
                        Z = Z + biases[ik];

                    if (bn == 'T')
                        Z = (((Z - running_mean[ik]) * inv_std[ik]) * gamma[ik]) + beta[ik];

                    if (relu == 'T')
                        Z = _mm_max_ps(Z, zeros);

                    hh = ih * s;
                    ww = iw * s;
                    // Yw[n, k, hh:hh+m, ww:ww+m] = Z[:min(m, H-hh), :min(m, W-ww)]
                    for (i = 0; i < min(m, ho - hh); i++)
                        for (j = 0; j < min(m, wo - ww); j++)
                            Yrow(in, ik, hh + i, ww + j) = Z[j * m + i];
                }
}

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_2x2_5x5_avx_fp32_nhwc
#else
void conv_winograd_2x2_5x5_avx_fp32_nchw
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
    conv_winograd_2x2_5x5_avx_fp32_nhwc_pre
#else
    conv_winograd_2x2_5x5_avx_fp32_nchw_pre
#endif
         (m, r, n, k, c, kh, kw, F, ldF1, ldF2, ldF3, U);

#ifdef TENSOR_FORMAT_NHWC
    conv_winograd_2x2_5x5_avx_fp32_nhwc_post
#else
    conv_winograd_2x2_5x5_avx_fp32_nchw_post
#endif
        (m, r, n, k, c, hi, wi, kh, kw, vpadding, hpadding,
         D, ldD1, ldD2, ldD3, Y, ldY1, ldY2, ldY3,
         biases, U, V, M, relu, bn, running_mean, inv_std,
         gamma, beta);
}

