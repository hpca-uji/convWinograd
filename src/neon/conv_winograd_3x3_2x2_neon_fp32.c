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

#include <arm_neon.h>
#include "neon_utils.h"

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
void conv_winograd_3x3_2x2_neon_fp32_nhwc_pre
#else
void conv_winograd_3x3_2x2_neon_fp32_nchw_pre
#endif
        (int m, int r, int n, int k, int c, int kh, int kw,
         float *F, int ldF1, int ldF2, int ldF3, float *U) {
    m = 3;
    r = 2;
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
    float32x2_t F0, F1,
            W0_, W1_, W2_, W3_,
            U0, U1, U2, U3,
            W0, W1, W2, W3;

    ldU3 = c;
    ldU2 = k * ldU3;
    ldU1 = t * ldU2;

#pragma omp parallel for collapse(2) private(ik, ic, F0, F1, W0, W1, W2, W3, W0_, W1_, W2_, W3_, U0, U1, U2, U3, i) if ((k * c) > 1)
    for (ik = 0; ik < k; ik++)
        for (ic = 0; ic < c; ic++) {
            // U[..., ik, ic] = (G @ F[ik, ic, ...]) @ G.T

            // Load rows of F: 2x2
            // The following solution is a bit "dirty" because F has 2 elements per row only,
            // but we load four to take advantage of vector instructions
            // This may generate a core dump if we try to access in an illegal position though.
            // The alternative is to load F2 scalar-wise. (There can be no problem with F0 and F1)
            for (j = 0; j < 2; j++) {
                F0[j] = Frow(ik, ic, 0, j);
                F1[j] = Frow(ik, ic, 1, j);
            }

            // We are doing extra flops here: each row has only 2 valid elements but we
            // use vector instructions that operate with 4 values each. For each row/vector register, the last entry
            // is actually garbage and, therefore, will not used in the subsequent "gemm", when accessing W
            // Wi  = G_row(i)  *  [ F0;F1;F2 ] (rows of F) with

            //   [     1,      0 ]     [ F01,  F01 ]     [ W00,  W01 ]
            //   [ 1./2.,  1./2. ]  *  [ F10,  F11 ]  =  [ W10,  W11 ]
            //   [ 1./2., -1./2. ]                       [ W20,  W21 ]
            //   [     0,      1 ]                       [ W30,  W31 ]
            W0_ = F0;
            W1_ = 0.5 * (F0 + F1);
            W2_ = W1_ - F1;
            W3_ = F1;

            // Transpose Wk so that
            // W0, W1 now contain the columns of the previous Wk
            //fvtrans_float32_4x4_neon_fp32( &W0, &W1, &W2, &W3 );
            W0[0] = W0_[0];
            W1[0] = W0_[1];
            W0[1] = W1_[0];
            W1[1] = W1_[1];
            W0[2] = W2_[0];
            W1[2] = W2_[1];
            W0[3] = W3_[0];
            W1[3] = W3_[1];

            //   [     1,      0 ]     [ W00,  W10,  W20,  W30 ]     [ U00,  U01,  U02,  U03 ]
            //   [ 1./2.,  1./2. ]  *  [ W01,  W11,  W21,  W31 ]  =  [ U10,  U11,  U12,  U13 ]
            //   [ 1./2., -1./2. ]                                   [ U20,  U21,  U22,  U23 ]
            //   [     0,      1 ]                                   [ U30,  U31,  U32,  U33 ]
            U0 = W0;
            U1 = 0.5 * (W0 + W1);
            U2 = U1 - W1;
            U3 = W1;

            // Scatter result in appropriate entries of U
            for (i = 0; i < 4; i++) {
                Urow(i, 0, ik, ic) = U0[i];
                Urow(i, 1, ik, ic) = U1[i];
                Urow(i, 2, ik, ic) = U2[i];
                Urow(i, 3, ik, ic) = U3[i];
            }
        }
}

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_3x3_2x2_neon_fp32_nhwc_kernel
#else
void conv_winograd_3x3_2x2_neon_fp32_nchw_kernel
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
    m = 3;
    r = 2;
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
    float32x4_t d0, d1, d2, d3,
            U0, U1, U2, U3,
            M0, M1, M2, M3,
            W0, W1, W2, W3,
            Z0, Z1, Z2,
            zeros = vmovq_n_f32(0.0);

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

#pragma omp parallel for collapse(2) private(ic, ih, hh_, hh, fh, oh, iw, ww_, ww, fw, ow, d0, d1, d2, d3, W0, W1, W2, W3, U0, U1, U2, U3, i, j) if ((n * c) > 1)
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

                    for (j = 0; j < t; j++) {
                        d0[j] = (fh <= 0 && 0 < oh && fw <= j && j < ow) ? Drow(in, ic, hh + 0 - fh, ww + j - fw) : 0.0;
                        d1[j] = (fh <= 1 && 1 < oh && fw <= j && j < ow) ? Drow(in, ic, hh + 1 - fh, ww + j - fw) : 0.0;
                        d2[j] = (fh <= 2 && 2 < oh && fw <= j && j < ow) ? Drow(in, ic, hh + 2 - fh, ww + j - fw) : 0.0;
                        d3[j] = (fh <= 3 && 3 < oh && fw <= j && j < ow) ? Drow(in, ic, hh + 3 - fh, ww + j - fw) : 0.0;
                    }

                    // Wi  = Bt_row(i)  *  [ d0;d1;d2;d3;d4,d5 ] (rows of d), with
                    //   [    1,    0,   -1,    0 ]     [ d00,  d01,  d02,  d03 ]     [ W00,  W01,  W02,  W03 ]
                    //   [    0,    1,    1,    0 ]  *  [ d10,  d11,  d12,  d13 ]  =  [ W10,  W11,  W12,  W13 ]
                    //   [    0,   -1,    1,    0 ]     [ d20,  d21,  d22,  d23 ]     [ W20,  W21,  W22,  W23 ]
                    //   [    0,   -1,    0,    1 ]     [ d30,  d31,  d32,  d33 ]     [ W30,  W31,  W32,  W33 ]
                    W0 = d0 - d2;
                    W1 = d1 + d2;
                    W2 = -d1 + d2;
                    W3 = -d1 + d3;

                    // Transpose Wk so that
                    // W0, W1, W2, W3 now contain the columns of the previous Wk
                    fvtrans_float32_4x4_neon_fp32(&W0, &W1, &W2, &W3);

                    // U_i  = Bt_row(i)  *  [ W0,W1,W2,W3 ] (rows of W/cols of W before transposition)
                    //   [    1,    0,   -1,    0 ]     [ W00,  W10,  W20,  W30 ]     [  U00,  U01,  U02,  U03 ]
                    //   [    0,    1,    1,    0 ]  *  [ W01,  W11,  W21,  W31 ]  =  [  U10,  U11,  U12,  U13 ]
                    //   [    0,   -1,    1,    0 ]     [ W02,  W12,  W22,  W32 ]     [  U20,  U21,  U22,  U23 ]
                    //   [    0,   -1,    0,    1 ]     [ W03,  W13,  W23,  W33 ]     [  U30,  U31,  U32,  U33 ]
                    U0 = W0 - W2;
                    U1 = W1 + W2;
                    U2 = -W1 + W2;
                    U3 = -W1 + W3;

                    // Scatter result in appropriate entries of V
                    for (i = 0; i < t; i++) {
                        Vrow(i, 0, ic, in * tile_h * tile_w + ih * tile_w + iw) = U0[i];
                        Vrow(i, 1, ic, in * tile_h * tile_w + ih * tile_w + iw) = U1[i];
                        Vrow(i, 2, ic, in * tile_h * tile_w + ih * tile_w + iw) = U2[i];
                        Vrow(i, 3, ic, in * tile_h * tile_w + ih * tile_w + iw) = U3[i];
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
#pragma omp parallel for collapse(2) private(in, ik, ih, iw, M0, M1, M2, M3, W0, W1, W2, W3, Z0, Z1, Z2, hh, ww, i, j) if ((n * k) > 1)
    for (in = 0; in < n; in++)
        for (ik = 0; ik < k; ik++)
            for (ih = 0; ih < tile_h; ih++)
                for (iw = 0; iw < tile_w; iw++) {
                    // Z = (At @ M[..., ik, in * tile_h * tile_w + ih * tile_w + iw]) @ At.T
                    // Take advantage that because of the change in the previous block of nested loops, M is now contiguous in memory.
                    // Therefore, we are actually computing the following:
                    //     Z = (At @ M[in * tile_h * tile_w + ih * tile_w + iw, ik, ...]) @ At.T

                    // Load rows of M: 4x4
                    for (i = 0; i < t; i++) {
                        M0[i] = Mrow(i, 0, ik, in * tile_h * tile_w + ih * tile_w + iw);
                        M1[i] = Mrow(i, 1, ik, in * tile_h * tile_w + ih * tile_w + iw);
                        M2[i] = Mrow(i, 2, ik, in * tile_h * tile_w + ih * tile_w + iw);
                        M3[i] = Mrow(i, 3, ik, in * tile_h * tile_w + ih * tile_w + iw);
                    }

                    // W_i  = A_row(i)  *  [ M0;M1;M2;M3 ] (rows of M), with
                    //    [      1,      1,      1,      0 ]     [ M00,  M01,  M02,  M03 ]     [ W00,  W01,  W02,  W03 ]
                    //    [      0,      1,     -1,      0 ]  *  [ M10,  M11,  M12,  M13 ]  =  [ W10,  W11,  W12,  W13 ]
                    //    [      0,      1,      1,      1 ]     [ M20,  M21,  M22,  M23 ]     [ W20,  W21,  W22,  W23 ]
                    //                                           [ M30,  M31,  M32,  M33 ]
                    W0 = M0 + M1 + M2;
                    W1 = M1 - M2;
                    W2 = M1 + M2 + M3;

                    // Transpose Wk so that
                    // W0, W1, W2, W3 now contain the columns of the previous Wk
                    fvtrans_float32_4x4_neon_fp32(&W0, &W1, &W2, &W3);

                    // In contrast with cases 1) and 2), in this case we do not use vector instructions for this second gemm as
                    // the result is only 2x2 and we would be doing many innecessary flops
                    //    [      1,      1,      1,      0 ]     [ W00,  W10,  W20 ]     [ Z00,  Z01,  Z02 ]
                    //    [      0,      1,     -1,      0 ]  *  [ W01,  W11,  W21 ]  =  [ Z10,  Z11,  Z12 ]
                    //    [      0,      1,      1,      1 ]     [ W02,  W12,  W22 ]     [ Z20,  Z21,  Z22 ]
                    //                                           [ W03,  W13,  W23 ]
                    Z0 = W0 + W1 + W2;
                    Z1 = W1 - W2;
                    Z2 = W1 + W2 + W3;

                    if (biases != NULL) {
                        Z0 = Z0 + biases[ik];
                        Z1 = Z1 + biases[ik];
                        Z2 = Z2 + biases[ik];
                    }

                    if (bn == 'T') {
                        Z0 = (((Z0 - running_mean[ik]) * inv_std[ik]) * gamma[ik]) + beta[ik];
                        Z1 = (((Z1 - running_mean[ik]) * inv_std[ik]) * gamma[ik]) + beta[ik];
                        Z2 = (((Z2 - running_mean[ik]) * inv_std[ik]) * gamma[ik]) + beta[ik];
                    }

                    if (relu == 'T') {
                        Z0 = vmaxq_f32(Z0, zeros);
                        Z1 = vmaxq_f32(Z1, zeros);
                        Z2 = vmaxq_f32(Z2, zeros);
                    }

                    hh = ih * s;
                    ww = iw * s;
                    // Yw[n, k, hh:hh+m, ww:ww+m] = Z[:min(m, H-hh), :min(m, W-ww)]
                    if (0 < min(m, ho - hh))
                        for (j = 0; j < min(m, wo - ww); j++)
                            Yrow(in, ik, hh + 0, ww + j) = Z0[j];
                    if (1 < min(m, ho - hh))
                        for (j = 0; j < min(m, wo - ww); j++)
                            Yrow(in, ik, hh + 1, ww + j) = Z1[j];
                    if (2 < min(m, ho - hh))
                        for (j = 0; j < min(m, wo - ww); j++)
                            Yrow(in, ik, hh + 2, ww + j) = Z2[j];
                }
}

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_3x3_2x2_neon_fp32_nhwc
#else
void conv_winograd_3x3_2x2_neon_fp32_nchw
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
    conv_winograd_3x3_2x2_neon_fp32_nhwc_pre
#else
    conv_winograd_3x3_2x2_neon_fp32_nchw_pre
#endif
         (m, r, n, k, c, kh, kw, F, ldF1, ldF2, ldF3, U);

#ifdef TENSOR_FORMAT_NHWC
    conv_winograd_3x3_2x2_neon_fp32_nhwc_kernel
#else
    conv_winograd_3x3_2x2_neon_fp32_nchw_kernel
#endif
        (m, r, n, k, c, hi, wi, kh, kw, vpadding, hpadding,
         D, ldD1, ldD2, ldD3, Y, ldY1, ldY2, ldY3,
         biases, U, V, M, relu, bn, running_mean, inv_std,
         gamma, beta);
}
