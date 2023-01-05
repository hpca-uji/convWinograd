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
#else
#define Drow(a1, a2, a3, a4)  D[ (a1)*(ldD1)+(a2)*(ldD2)+(a3)*(ldD3)+(a4) ]
#define Frow(a1, a2, a3, a4)  F[ (a1)*(ldF1)+(a2)*(ldF2)+(a3)*(ldF3)+(a4) ]
#define Yrow(a1, a2, a3, a4)  Y[ (a1)*(ldY1)+(a2)*(ldY2)+(a3)*(ldY3)+(a4) ]
#endif

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_2x2_3x3_native_fp32_nhwc_pre
#else
void conv_winograd_2x2_3x3_native_fp32_nchw_pre
#endif
        (int m, int r, int n, int k, int c, int kh, int kw,
         float *F, int ldF1, int ldF2, int ldF3, float *U) {
    m = 2;
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
    float W[3][4];

    ldU3 = c;
    ldU2 = k * ldU3;
    ldU1 = t * ldU2;

#ifdef DEBUG
    double      t1, t2, T1;
    t1 = dclock();
#endif

#pragma omp parallel for collapse(2) private(ik, ic, W, j) if ((k * c) > 1)
    for (ik = 0; ik < k; ik++)
        for (ic = 0; ic < c; ic++) {
            // Wi  = G_row(i)  *  [ F0;F1;F2 ] (rows of F) with
            // G = [1.0,  0.0, 0.0,
            //      0.5,  0.5, 0.5,
            //      0.5, -0.5, 0.5,
            //      0.0,  0.0, 1.0];
            for (j = 0; j < 3; j++) {
                W[j][0] = Frow(ik, ic, 0, j);
                W[j][1] = 0.5 * (Frow(ik, ic, 0, j) + Frow(ik, ic, 1, j) + Frow(ik, ic, 2, j));
                W[j][2] = 0.5 * (Frow(ik, ic, 0, j) - Frow(ik, ic, 1, j) + Frow(ik, ic, 2, j));
                W[j][3] = Frow(ik, ic, 2, j);
            }
            // Ui  = G_row(i)  *  [ W0,W1,W2 ] (rows of W/cols of W before transposition)
            for (j = 0; j < 4; j++) {
                Urow(j, 0, ik, ic) = W[0][j];
                Urow(j, 1, ik, ic) = 0.5 * (W[0][j] + W[1][j] + W[2][j]);
                Urow(j, 2, ik, ic) = 0.5 * (W[0][j] - W[1][j] + W[2][j]);
                Urow(j, 3, ik, ic) = W[2][j];
            }
        }

#ifdef DEBUG
    t2 = dclock();
    T1 = t2 - t1;
    float tot = T1;
    printf("%12.8f\n", T1/tot*100);
#endif
}

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_2x2_3x3_native_fp32_nhwc_kernel
#else
void conv_winograd_2x2_3x3_native_fp32_nchw_kernel
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
            i, j, ho, wo, e, v;

    float W[4][4], Z[4];
#ifdef DEBUG
    double      t1, t2, T2, T3, T4;
#endif

    ho = (hi + 2 * vpadding - kh) / vstride + 1;
    wo = (wi + 2 * hpadding - kw) / hstride + 1;

    tile_h = ceil(((double) hi + 2 * vpadding - t) / s) + 1;
    tile_w = ceil(((double) wi + 2 * hpadding - t) / s) + 1;

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

#pragma omp parallel for collapse(2) private(ic, ih, hh_, hh, fh, oh, iw, ww_, ww, fw, ow, W, j) if ((n * c) > 1)
    for (in = 0; in < n; in++)
        for (ic = 0; ic < c; ic++)
            for (ih = 0; ih < tile_h; ih++) {
                hh_ = min(hi, ih * s - vpadding);
                hh = max(hh_, 0);
                fh = min(max(-hh_, 0), t);
                oh = max(min(hi - hh, t), 0);
                oh = oh < t ? oh + fh : oh;

                for (iw = 0; iw < tile_w; iw++) {
                    ww_ = min(wi, iw * s - hpadding);
                    ww = max(ww_, 0);
                    fw = min(max(-ww_, 0), t);
                    ow = max(min(wi - ww, t), 0);
                    ow = ow < t ? ow + fw : ow;

#define drow(ii, jj) ((fh <= ii && ii < oh && fw <= jj && jj < ow) ? Drow(in, ic, hh + ii - fh, ww + jj - fw) : 0.0)

                    // Wi  = Bt_row(i)  *  [ d0;d1;d2;d3 ] (rows of d), with
                    // Bt = [1.0,  0.0, -1.0,  0.0,
                    //       0.0,  1.0,  1.0,  0.0,
                    //       0.0, -1.0,  1.0,  0.0,
                    //       0.0,  1.0,  0.0, -1.0];
                    for (j = 0; j < 4; j++) {
                        W[j][0] =  drow(0, j) - drow(2, j);
                        W[j][1] =  drow(1, j) + drow(2, j);
                        W[j][2] = -drow(1, j) + drow(2, j);
                        W[j][3] =  drow(1, j) - drow(3, j);
                    }

                    // U_i  = Bt_row(i)  *  [ W0,W1,W2,W3 ] (rows of W/cols of W before transposition)
                    for (j = 0; j < 4; j++) {
                        Vrow(j, 0, ic, in * tile_h * tile_w + ih * tile_w + iw) =  W[0][j] - W[2][j];
                        Vrow(j, 1, ic, in * tile_h * tile_w + ih * tile_w + iw) =  W[1][j] + W[2][j];
                        Vrow(j, 2, ic, in * tile_h * tile_w + ih * tile_w + iw) = -W[1][j] + W[2][j];
                        Vrow(j, 3, ic, in * tile_h * tile_w + ih * tile_w + iw) =  W[1][j] - W[3][j];
                    }
                }
            }

#ifdef DEBUG
    t2 = dclock();
    T2 = t2 - t1;
    t1 = dclock();
#endif

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

#pragma omp parallel for collapse(2) private(in, ik, ih, iw, W, Z) if ((n * k) > 1)
    for (in = 0; in < n; in++)
        for (ik = 0; ik < k; ik++)
            for (ih = 0; ih < tile_h; ih++)
                for (iw = 0; iw < tile_w; iw++) {
                    // Z = (At @ M[..., ik, in * tile_h * tile_w + ih * tile_w + iw]) @ At.T
                    // Take advantage that because of the change in the previous block of nested loops, M is now contiguous in memory.
                    // Therefore, we are actually computing the following:
                    //     Z = (At @ M[in * tile_h * tile_w + ih * tile_w + iw, ik, ...]) @ At.T

#define mrow(ii, jj) Mrow(ii, jj, ik, in * tile_h * tile_w + ih * tile_w + iw)

                    // Load rows of M: 4x4
                    for (j = 0; j < 4; j++) {
                        W[0][j] = mrow(j, 0) + mrow(j, 1) + mrow(j, 2);
                        W[1][j] = mrow(j, 1) - mrow(j, 2) - mrow(j, 3);
                    }

                    // In contrast with cases 1) and 2), in this case we do not use vector instructions for this second gemm as
                    // the result is only 2x2 and we would be doing many innecessary flops
                    Z[0] = W[0][0] + W[0][1] + W[0][2];
                    Z[1] = W[0][1] - W[0][2] - W[0][3];
                    Z[2] = W[1][0] + W[1][1] + W[1][2];
                    Z[3] = W[1][1] - W[1][2] - W[1][3];

                    if (biases != NULL)
                        for (j = 0; j < 4; j++)
                            Z[j] = Z[j] + biases[ik];

                    if (bn == 'T')
                        for (j = 0; j < 4; j++)
                            Z[j] = (((Z[j] - running_mean[ik]) * inv_std[ik]) * gamma[ik]) + beta[ik];

                    if (relu == 'T')
                        for (j = 0; j < 4; j++)
                            Z[j] = Z[j] > 0 ? Z[j] : 0.0;

                    hh = ih * s;
                    ww = iw * s;
                    // Yw[n, k, hh:hh+m, ww:ww+m] = Z[:min(m, H-hh), :min(m, W-ww)]
                    for (i = 0; i < min(m, ho - hh); i++)
                        for (j = 0; j < min(m, wo - ww); j++)
                            Yrow(in, ik, hh + i, ww + j) = Z[j * m + i];
                }

#ifdef DEBUG
    t2 = dclock();
    T4 = t2 - t1;
    float tot = T2 + T3 + T4;
    printf("%12.8f %12.8f %12.8f \n", T2/tot*100, T3/tot*100, T4/tot*100);
#endif
}

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_2x2_3x3_native_fp32_nhwc
#else
void conv_winograd_2x2_3x3_native_fp32_nchw
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
    conv_winograd_2x2_3x3_native_fp32_nhwc_pre
#else
    conv_winograd_2x2_3x3_native_fp32_nchw_pre
#endif
        (m, r, n, k, c, kh, kw, F, ldF1, ldF2, ldF3, U);

#ifdef TENSOR_FORMAT_NHWC
    conv_winograd_2x2_3x3_native_fp32_nhwc_kernel
#else
    conv_winograd_2x2_3x3_native_fp32_nchw_kernel
#endif
        (m, r, n, k, c, hi, wi, kh, kw, vpadding, hpadding,
         D, ldD1, ldD2, ldD3, Y, ldY1, ldY2, ldY3,
         biases, U, V, M, relu, bn, running_mean, inv_std,
         gamma, beta);
}
