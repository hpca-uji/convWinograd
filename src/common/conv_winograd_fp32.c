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

#define min(a, b)     ( (a) > (b) ? (b) : (a) )
#define max(a, b)     ( (a) > (b) ? (a) : (b) )

#define Urow(a1, a2, a3, a4)  U[ (a1)*(ldU1)+(a2)*(ldU2)+(a3)*(ldU3)+(a4) ]
#define Vrow(a1, a2, a3, a4)  V[ (a1)*(ldV1)+(a2)*(ldV2)+(a3)*(ldV3)+(a4) ]
#define Mrow(a1, a2, a3, a4)  M[ (a1)*(ldM1)+(a2)*(ldM2)+(a3)*(ldM3)+(a4) ]

#ifdef TENSOR_FORMAT_NHWC
#define Drow(a1,a2,a3,a4)  D[ (a1)*(ldD1)+(a3)*(ldD2)+(a4)*(ldD3)+(a2) ]
#define Frow(a1,a2,a3,a4)  F[ (a2)*(ldF1)+(a3)*(ldF2)+(a4)*(ldF3)+(a1) ]
#define Yrow(a1,a2,a3,a4)  Y[ (a1)*(ldY1)+(a3)*(ldY2)+(a4)*(ldY3)+(a2) ]
void conv_winograd_fp32_nhwc
#else
#define Drow(a1, a2, a3, a4)  D[ (a1)*(ldD1)+(a2)*(ldD2)+(a3)*(ldD3)+(a4) ]
#define Frow(a1, a2, a3, a4)  F[ (a1)*(ldF1)+(a2)*(ldF2)+(a3)*(ldF3)+(a4) ]
#define Yrow(a1, a2, a3, a4)  Y[ (a1)*(ldY1)+(a2)*(ldY2)+(a3)*(ldY3)+(a4) ]

void conv_winograd_fp32_nchw
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
            i, j, ho, wo,
            e, v;
    // float d[t*t], Wk[t*t], Uk[t*t], Z[m*m];

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

#pragma omp parallel for collapse(2) private(ik, ic)
    for (ik = 0; ik < k; ik++)
        for (ic = 0; ic < c; ic++) {
            // U[..., ik, ic] = (G @ F[ik, ic, ...]) @ G.T
            float Wk[t * t], Uk[t * t];
#if CBLAS_TYPE_CBLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  t, r, r,
                  1.0, G, r,
                       &Frow(ik, ic, 0, 0), r,
                  0.0, Wk, r );
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                  t, t, r,
                  1.0, Wk, r,
                       G,  r,
                  0.0, Uk, t );
#else
            gemm('R', 'R', 'R',
                 'N', 'N',
                 t, r, r,
                 1.0, G, r,
                 &Frow(ik, ic, 0, 0), r,
                 0.0, Wk, r);
            gemm('R', 'R', 'R',
                 'N', 'T',
                 t, t, r,
                 1.0, Wk, r,
                 G, r,
                 0.0, Uk, t);
#endif
            for (i = 0; i < t; i++)
                for (j = 0; j < t; j++)
                    Urow(i, j, ik, ic) = Uk[i * t + j];
        }
#pragma omp parallel for collapse(2) private(ic, ih, hh_, hh, fh, oh, iw, ww_, ww, fw, ow, i, j)
    for (in = 0; in < n; in++)
        for (ic = 0; ic < c; ic++)
            for (ih = 0; ih < tile_h; ih++) {
                hh_ = min(hi, ih * s - vpadding);
                hh = max(hh_, 0);
                fh = min(max(-hh_, 0), t);
                oh = max(min(hi - hh, t), 0);

                for (iw = 0; iw < tile_w; iw++) {
                    ww_ = min(wi, iw * s - hpadding);
                    ww = max(ww_, 0);
                    fw = min(max(-ww_, 0), t);
                    ow = max(min(wi - ww, t), 0);

                    float Wk[t * t], Uk[t * t];
                    for (i = 0; i < t; i++)
                        for (j = 0; j < t; j++)
                            Uk[i * t + j] = ((fh <= i && i < oh && fw <= j && j < ow) ? Drow(in, ic, hh + i - fh,
                                                                                             ww + j - fw) : 0.0);

                    // V[..., ic, in * tile_h * tile_w + ih * tile_w + iw] = (Bt @ d) @ Bt.T
#if CBLAS_TYPE_CBLAS
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                          t, t, t,
                          1.0, Bt, t,
                               Uk, t,
                          0.0, Wk, t );
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                          t, t, t,
                          1.0, Wk, t,
                               Bt, t,
                          0.0, Uk, t );
#else
                    gemm('R', 'R', 'R',
                         'N', 'N',
                         t, t, t,
                         1.0, Bt, t,
                         Uk, t,
                         0.0, Wk, t);
                    gemm('R', 'R', 'R',
                         'N', 'T',
                         t, t, t,
                         1.0, Wk, t,
                         Bt, t,
                         0.0, Uk, t);
#endif
                    for (i = 0; i < t; i++)
                        for (j = 0; j < t; j++)
                            Vrow(i, j, ic, in * tile_h * tile_w + ih * tile_w + iw) = Uk[i * t + j];
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
#pragma omp parallel for collapse(2) private(in, ik, ih, iw, hh, ww, i, j)
    for (in = 0; in < n; in++)
        for (ik = 0; ik < k; ik++)
            for (ih = 0; ih < tile_h; ih++)
                for (iw = 0; iw < tile_w; iw++) {
                    // Z = (At @ M[..., ik, in * tile_h * tile_w + ih * tile_w + iw]) @ At.T
                    // Take advantage that because of the change in the previous block of nested loops, M is now contiguous in memory.
                    // Therefore, we are actually computing the following:
                    //     Z = (At @ M[in * tile_h * tile_w + ih * tile_w + iw, ik, ...]) @ At.T
                    float Wk[t * t], Uk[t * t], Z[m * m];
                    for (i = 0; i < t; i++)
                        for (j = 0; j < t; j++)
                            Uk[j * t + i] = Mrow(i, j, ik, in * tile_h * tile_w + ih * tile_w + iw);
#if CBLAS_TYPE_CBLAS
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                          m, t, t,
                          1.0, At, t,
                               Uk, t,
                          0.0, Wk, t );
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                          m, m, t,
                          1.0, Wk, t,
                               At, t,
                          0.0, Z, m );
#else
                    gemm('R', 'R', 'R',
                         'N', 'N',
                         m, t, t,
                         1.0, At, t,
                         Uk, t,
                         0.0, Wk, t);
                    gemm('R', 'R', 'R',
                         'N', 'T',
                         m, m, t,
                         1.0, Wk, t,
                         At, t,
                         0.0, Z, m);
#endif
                    hh = ih * s;
                    ww = iw * s;
                    // Yw[n, k, hh:hh+m, ww:ww+m] = Z[:min(m, H-hh), :min(m, W-ww)]
                    for (i = 0; i < min(m, ho - hh); i++)
                        for (j = 0; j < min(m, wo - ww); j++) {
                            Yrow(in, ik, hh + i, ww + j) = Z[j * m + i];
                            if (biases != NULL)
                                Yrow(in, ik, hh + i, ww + j) += biases[ik];
                            if (bn == 'T')
                                Yrow(in, ik, hh + i, ww + j) =
                                        (((Yrow(in, ik, hh + i, ww + j) - running_mean[ik]) * inv_std[ik]) *
                                         gamma[ik]) + beta[ik];
                            if (relu == 'T')
                                Yrow(in, ik, hh + i, ww + j) = max(Yrow(in, ik, hh + i, ww + j), 0);
                        }
                }
}
