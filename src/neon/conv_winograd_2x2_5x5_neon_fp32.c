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
#define drow(a1,a2)        (( fh <= a1 && a1 < oh && fw <= a2 && a2 < ow ) ? Drow(in, ic, hh + a1 - fh, ww + a2 - fw) : 0.0)
#define Mprow(a1,a2)       Mrow(a2, a1, ik, in * tile_h * tile_w + ih * tile_w + iw)
#else
#define Drow(a1, a2, a3, a4)  D[ (a1)*(ldD1)+(a2)*(ldD2)+(a3)*(ldD3)+(a4) ]
#define Frow(a1, a2, a3, a4)  F[ (a1)*(ldF1)+(a2)*(ldF2)+(a3)*(ldF3)+(a4) ]
#define Yrow(a1, a2, a3, a4)  Y[ (a1)*(ldY1)+(a2)*(ldY2)+(a3)*(ldY3)+(a4) ]
#define drow(a1, a2)       (( fh <= a1 && a1 < oh && fw <= a2 && a2 < ow ) ? Drow(in, ic, hh + a1 - fh, ww + a2 - fw) : 0.0)
#define Mprow(a1, a2)      Mrow(a2, a1, ik, in * tile_h * tile_w + ih * tile_w + iw)
#endif

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_2x2_5x5_neon_fp32_nhwc_pre
#else
void conv_winograd_2x2_5x5_neon_fp32_nchw_pre
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
    float   U04, U14, U24, U34, U44, U54,
            U05, U15, U25, U35, U45, U55,
            W44, W45, W54, W55,
            W04, W05, W14, W15;
    float32x4_t F0, F1, F2, F3, F4,
            U0, U1, U2, U3, U4, U5,
            W0, W1, W2, W3, W4, W5,
            W4_, W5_;

    ldU3 = c;
    ldU2 = k * ldU3;
    ldU1 = t * ldU2;

#pragma omp parallel for collapse(2) private(ik, ic, F0, F1, F2, F3, F4, W0, W1, W2, W3, W4, W5, W4_, W44, W54, U0, U1, U2, U3, U4, U5, U04, U05, U14, U15, U24, U25, U34, U35, U44, U45, U54, U55, i) if ((k * c) > 1)
    for (ik = 0; ik < k; ik++)
        for (ic = 0; ic < c; ic++) {
            // U[..., ik, ic] = (G @ F[ik, ic, ...]) @ G.T

            // Load rows of F: 3x3
            // The following solution is a bit "dirty" because F has 3 elements per row only,
            // but we load four to take advantage of vector instructions
            // This may generate a core dump if we try to access in an illegal position though.
            // The alternative is to load F2 scalar-wise. (There can be no problem with F0 and F1)
            for (j = 0; j < 4; j++) {
                F0[j] = Frow(ik, ic, 0, j);
                F1[j] = Frow(ik, ic, 1, j);
                F2[j] = Frow(ik, ic, 2, j);
                F3[j] = Frow(ik, ic, 3, j);
                F4[j] = Frow(ik, ic, 4, j);
            }

            // We are doing extra flops here: each row has only 3 valid elements but we
            // use vector instructions that operate with 4 values each. For each row/vector register, the last entry
            // is actually garbage and, therefore, will not used in the subsequent "gemm", when accessing W
            // Wi  = G_row(i)  *  [ F0;F1;F2 ] (rows of F) with

            //   [  1./4.,      0,      0,      0,      0   ]     [ F01,  F01,  F02,  F03 | F04]     [W00,  W01,  W02,  W03 | W04]
            //   [ -1./6., -1./6., -1./6., -1./6., -1./6.   ]  *  [ F10,  F11,  F12,  F13 | F14]  =  [W10,  W11,  W12,  W13 | W14]
            //   [ -1./6.,  1./6., -1./6.,  1./6., -1./6.   ]     [ F20,  F21,  F22,  F23 | F24]     [W20,  W21,  W22,  W23 | W24]
            //   [ 1./24., 1./12.,  1./6.,  1./3.,  2./3.   ]     [ F30,  F31,  F32,  F33 | F44]     [W30,  W31,  W32,  W33 | W34]
            //   [ 1./24.,-1./12.,  1./6., -1./3.,  2./3.   ]     [ F40,  F41,  F42,  F43 | F44]     [W40,  W41,  W42,  W43 | W44]
            //   [      0,      0,      0,      0,      1   ]                                        [W50,  W51,  W52,  W53 | W54]
            W0 = (float) (1.0 / 4.0) * F0;
            W4_[0] = 1.0 / 4.0 * Frow(ik, ic, 0, 4);

            W1 = (float) (1.0 / 6.0) * (-F0 - F1 - F2 - F3 - F4);
            W4_[1] = 1.0 / 6.0 * (-Frow(ik, ic, 0, 4) - Frow(ik, ic, 1, 4) - Frow(ik, ic, 2, 4) - Frow(ik, ic, 3, 4) -
                                  Frow(ik, ic, 4, 4));

            W2 = W1 + (float) (2.0 / 6.0) * (F1 + F3);
            W4_[2] = W4_[1] + 2.0 / 6.0 * (Frow(ik, ic, 1, 4) + Frow(ik, ic, 3, 4));

            W3 = (float) (1.0 / 24.0) * F0 + (float) (1.0 / 12.0) * F1 + (float) (1.0 / 6.0) * F2 +
                 (float) (1.0 / 3.0) * F3 + (float) (2.0 / 3.0) * F4;
            W4_[3] =
                    1.0 / 24.0 * Frow(ik, ic, 0, 4) + 1.0 / 12.0 * Frow(ik, ic, 1, 4) + 1.0 / 6.0 * Frow(ik, ic, 2, 4) +
                    1.0 / 3.0 * Frow(ik, ic, 3, 4) + 2.0 / 3.0 * Frow(ik, ic, 4, 4);

            W4 = W3 - (float) (2.0 / 12.0) * F1 - (float) (2.0 / 3.0) * F3;
            W44 = W4_[3] - 2.0 / 12.0 * Frow(ik, ic, 1, 4) - 2.0 / 3.0 * Frow(ik, ic, 3, 4);

            W5 = F4;
            W54 = Frow(ik, ic, 4, 4);

            // Transpose Wk so that
            // W0, W1, W2, W3 now contain the columns of the previous Wk
            // Note that, after the transposition, W3 contains garbage
            // and it will not be used in the subsequent operations
            // The transposition of W4 and W5 are not performed as the register only would have 2 valid entries
            fvtrans_float32_4x4_neon_fp32(&W0, &W1, &W2, &W3);

            //   [  1./4.,      0,      0,      0,      0   ]     [ W00,  W10,  W20,  W30 | W40,  W50]     [ U00,  U01,  U02,  U03 | U04,  U05 ]
            //   [ -1./6., -1./6., -1./6., -1./6., -1./6.   ]  *  [ W01,  W11,  W21,  W31 | W41,  W51]  =  [ U10,  U11,  U12,  U13 | U14,  U15 ]
            //   [ -1./6.,  1./6., -1./6.,  1./6., -1./6.   ]     [ W02,  W12,  W22,  W32 | W42,  W52]     [ U20,  U21,  U22,  U23 | U24,  U25 ]
            //   [ 1./24., 1./12.,  1./6.,  1./3.,  2./3.   ]     [ W03,  W13,  W23,  W33 | W43,  W53]     [ U30,  U31,  U32,  U33 | U34,  U35 ]
            //   [ 1./24.,-1./12.,  1./6., -1./3.,  2./3.   ]     [ W04,  W14,  W24,  W34 | W44,  W54]     [ U40,  U41,  U42,  U43 | U44,  U45 ]
            //   [      0,      0,      0,      0,      1   ]                                              [ U50,  U51,  U52,  U53 | U54,  U55 ]
            U0 = (float) (1.0 / 4.0) * W0;
            U04 = 1.0 / 4.0 * W4[0];
            U05 = 1.0 / 4.0 * W5[0];

            U1 = (float) (1.0 / 6.0) * (-W0 - W1 - W2 - W3 - W4_);
            U14 = 1.0 / 6.0 * (-W4[0] - W4[1] - W4[2] - W4[3] - W44);
            U15 = 1.0 / 6.0 * (-W5[0] - W5[1] - W5[2] - W5[3] - W54);

            U2 = U1 + (float) (2.0 / 6.0) * (W1 + W3);
            U24 = U14 + 2.0 / 6.0 * (W4[1] + W4[3]);
            U25 = U15 + 2.0 / 6.0 * (W5[1] + W5[3]);

            U3 = (float) (1.0 / 24.0) * W0 + (float) (1.0 / 12.0) * W1 + (float) (1.0 / 6.0) * W2 +
                 (float) (1.0 / 3.0) * W3 + (float) (2.0 / 3.0) * W4_;
            U34 = 1.0 / 24.0 * W4[0] + 1.0 / 12.0 * W4[1] + 1.0 / 6.0 * W4[2] + 1.0 / 3.0 * W4[3] + 2.0 / 3.0 * W44;
            U35 = 1.0 / 24.0 * W5[0] + 1.0 / 12.0 * W5[1] + 1.0 / 6.0 * W5[2] + 1.0 / 3.0 * W5[3] + 2.0 / 3.0 * W54;

            U4 = U3 - (float) (2.0 / 12.0) * W1 - (float) (2.0 / 3.0) * W3;
            U44 = U34 - 2.0 / 12.0 * W4[1] - 2.0 / 3.0 * W4[3];
            U45 = U35 - 2.0 / 12.0 * W5[1] - 2.0 / 3.0 * W5[3];

            U5 = W4_;
            U54 = W44;
            U55 = W54;

            // Scatter result in appropriate entries of U
            for (i = 0; i < 4; i++) {
                Urow(i, 0, ik, ic) = U0[i];
                Urow(i, 1, ik, ic) = U1[i];
                Urow(i, 2, ik, ic) = U2[i];
                Urow(i, 3, ik, ic) = U3[i];
                Urow(i, 4, ik, ic) = U4[i];
                Urow(i, 5, ik, ic) = U5[i];
            }

            Urow(4, 0, ik, ic) = U04;
            Urow(4, 1, ik, ic) = U14;
            Urow(4, 2, ik, ic) = U24;
            Urow(4, 3, ik, ic) = U34;
            Urow(4, 4, ik, ic) = U44;
            Urow(4, 5, ik, ic) = U54;

            Urow(5, 0, ik, ic) = U05;
            Urow(5, 1, ik, ic) = U15;
            Urow(5, 2, ik, ic) = U25;
            Urow(5, 3, ik, ic) = U35;
            Urow(5, 4, ik, ic) = U45;
            Urow(5, 5, ik, ic) = U55;
        }
}

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_2x2_5x5_neon_fp32_nhwc_kernel
#else
void conv_winograd_2x2_5x5_neon_fp32_nchw_kernel
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
            i, j, ho, wo, e, v;
    float   U04, U14, U24, U34, U44, U54,
            U05, U15, U25, U35, U45, U55,
            W44, W45, W54, W55,
            W04, W05, W14, W15;
    float32x4_t d0, d1, d2, d3, d4, d5,
            U0, U1, U2, U3, U4, U5,
            M0, M1, M2, M3, M4, M5,
            W0, W1, W2, W3, W4, W5,
            Z, W4_, W5_,
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

#pragma omp parallel for collapse(2) private(ic, ih, hh_, hh, fh, oh, iw, ww_, ww, fw, ow, d0, d1, d2, d3, d4, d5, W0, W1, W2, W3, W4, W5, W4_, W5_, U0, U1, U2, U3, U4, U5, U04, U05, U14, U15, U24, U25, U34, U35, U44, U45, U54, U55, i, j) if ((n * c) > 1)
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

                    for (j = 0; j < 4; j++) {
                        d0[j] = (fh <= 0 && 0 < oh && fw <= j && j < ow) ? Drow(in, ic, hh + 0 - fh, ww + j - fw) : 0.0;
                        d1[j] = (fh <= 1 && 1 < oh && fw <= j && j < ow) ? Drow(in, ic, hh + 1 - fh, ww + j - fw) : 0.0;
                        d2[j] = (fh <= 2 && 2 < oh && fw <= j && j < ow) ? Drow(in, ic, hh + 2 - fh, ww + j - fw) : 0.0;
                        d3[j] = (fh <= 3 && 3 < oh && fw <= j && j < ow) ? Drow(in, ic, hh + 3 - fh, ww + j - fw) : 0.0;
                        d4[j] = (fh <= 4 && 4 < oh && fw <= j && j < ow) ? Drow(in, ic, hh + 4 - fh, ww + j - fw) : 0.0;
                        d5[j] = (fh <= 5 && 5 < oh && fw <= j && j < ow) ? Drow(in, ic, hh + 5 - fh, ww + j - fw) : 0.0;
                    }

                    // Wi  = Bt_row(i)  *  [ d0;d1;d2;d3;d4,d5 ] (rows of d), with
                    //   [    4,    0,   -5,    0,    1,    0 ]     [  d00,  d01,  d02,  d03 | d04,  d05 ]     [  W00,  W01,  W02,  W03 | W04,  W05 ]
                    //   [    0,   -4,   -4,    1,    1,    0 ]     [  d10,  d11,  d12,  d13 | d14,  d15 ]     [  W10,  W11,  W12,  W13 | W14,  W15 ]
                    //   [    0,    4,   -4,   -1,    1,    0 ]  *  [  d20,  d21,  d22,  d23 | d24,  d25 ]  =  [  W20,  W21,  W22,  W23 | W24,  W25 ]
                    //   [    0,   -2,   -1,    2,    1,    0 ]     [  d30,  d31,  d32,  d33 | d34,  d35 ]     [  W30,  W31,  W32,  W33 | W34,  W35 ]
                    //   [    0,    2,   -1,   -2,    1,    0 ]     [  d40,  d41,  d42,  d43 | d44,  d45 ]     [  W40,  W41,  W42,  W43 | W44,  W45 ]
                    //   [    0,    4,    0,   -5,    0,    1 ]     [  d50,  d51,  d52,  d53 | d54,  d55 ]     [  W50,  W51,  W52,  W53 | W54,  W55 ]
                    W0 = 4.0 * d0 - 5.0 * d2 + d4;
                    W4_[0] = 4.0 * drow(0, 4) - 5.0 * drow(2, 4) + drow(4, 4);
                    W5_[0] = 4.0 * drow(0, 5) - 5.0 * drow(2, 5) + drow(4, 5);

                    W1 = 4.0 * (-d1 - d2) + d3 + d4;
                    W4_[1] = 4.0 * (-drow(1, 4) - drow(2, 4)) + drow(3, 4) + drow(4, 4);
                    W5_[1] = 4.0 * (-drow(1, 5) - drow(2, 5)) + drow(3, 5) + drow(4, 5);

                    W2 = 4.0 * (d1 - d2) - d3 + d4;
                    W4_[2] = 4.0 * (drow(1, 4) - drow(2, 4)) - drow(3, 4) + drow(4, 4);
                    W5_[2] = 4.0 * (drow(1, 5) - drow(2, 5)) - drow(3, 5) + drow(4, 5);

                    W3 = 2.0 * (-d1 + d3) - d2 + d4;
                    W4_[3] = 2.0 * (-drow(1, 4) + drow(3, 4)) - drow(2, 4) + drow(4, 4);
                    W5_[3] = 2.0 * (-drow(1, 5) + drow(3, 5)) - drow(2, 5) + drow(4, 5);

                    W4 = 2.0 * (d1 - d3) - d2 + d4;
                    W44 = 2.0 * (drow(1, 4) - drow(3, 4)) - drow(2, 4) + drow(4, 4);
                    W45 = 2.0 * (drow(1, 5) - drow(3, 5)) - drow(2, 5) + drow(4, 5);

                    W5 = 4.0 * d1 - 5.0 * d3 + d5;
                    W54 = 4.0 * drow(1, 4) - 5.0 * drow(3, 4) + drow(5, 4);
                    W55 = 4.0 * drow(1, 5) - 5.0 * drow(3, 5) + drow(5, 5);

                    // Transpose Wk so that
                    // W0, W1, W2, W3 now contain the columns of the previous Wk
                    fvtrans_float32_4x4_neon_fp32(&W0, &W1, &W2, &W3);

                    // U_i  = Bt_row(i)  *  [ W0,W1,W2,W3 ] (rows of W/cols of W before transposition)
                    //   [    4,    0,   -5,    0,    1,    0 ]     [  W00,  W10,  W20,  W30 | W40,  W50 ]     [  U00,  U01,  U02,  U03 | U04,  U05 ]
                    //   [    0,   -4,   -4,    1,    1,    0 ]     [  W01,  W11,  W21,  W31 | W41,  W51 ]     [  U10,  U11,  U12,  U13 | U14,  U15 ]
                    //   [    0,    4,   -4,   -1,    1,    0 ]  *  [  W02,  W12,  W22,  W32 | W42,  W52 ]  =  [  U20,  U21,  U22,  U23 | U24,  U25 ]
                    //   [    0,   -2,   -1,    2,    1,    0 ]     [  W03,  W13,  W23,  W33 | W43,  W53 ]     [  U30,  U31,  U32,  U33 | U34,  U35 ]
                    //   [    0,    2,   -1,   -2,    1,    0 ]     [  W04,  W14,  W24,  W34 | W44,  W54 ]     [  U40,  U41,  U42,  U43 | U44,  U45 ]
                    //   [    0,    4,    0,   -5,    0,    1 ]     [  W05,  W15,  W25,  W35 | W45,  W55 ]     [  U50,  U51,  U52,  U53 | U54,  U55 ]
                    U0 = 4.0 * W0 - 5.0 * W2 + W4_;
                    U04 = 4.0 * W4[0] - 5.0 * W4[2] + W44;
                    U05 = 4.0 * W5[0] - 5.0 * W5[2] + W54;

                    U1 = 4.0 * (-W1 - W2) + W3 + W4_;
                    U14 = 4.0 * (-W4[1] - W4[2]) + W4[3] + W44;
                    U15 = 4.0 * (-W5[1] - W5[2]) + W5[3] + W54;

                    U2 = 4.0 * (W1 - W2) - W3 + W4_;
                    U24 = 4.0 * (W4[1] - W4[2]) - W4[3] + W44;
                    U25 = 4.0 * (W5[1] - W5[2]) - W5[3] + W54;

                    U3 = 2.0 * (-W1 + W3) - W2 + W4_;
                    U34 = 2.0 * (-W4[1] + W4[3]) - W4[2] + W44;
                    U35 = 2.0 * (-W5[1] + W5[3]) - W5[2] + W54;

                    U4 = 2.0 * (W1 - W3) - W2 + W4_;
                    U44 = 2.0 * (W4[1] - W4[3]) - W4[2] + W44;
                    U45 = 2.0 * (W5[1] - W5[3]) - W5[2] + W54;

                    U5 = 4.0 * W1 - 5.0 * W3 + W5_;
                    U54 = 4.0 * W4[1] - 5.0 * W4[3] + W45;
                    U55 = 4.0 * W5[1] - 5.0 * W5[3] + W55;

                    // Scatter result in appropriate entries of V
                    for (i = 0; i < 4; i++) {
                        Vrow(i, 0, ic, in * tile_h * tile_w + ih * tile_w + iw) = U0[i];
                        Vrow(i, 1, ic, in * tile_h * tile_w + ih * tile_w + iw) = U1[i];
                        Vrow(i, 2, ic, in * tile_h * tile_w + ih * tile_w + iw) = U2[i];
                        Vrow(i, 3, ic, in * tile_h * tile_w + ih * tile_w + iw) = U3[i];
                        Vrow(i, 4, ic, in * tile_h * tile_w + ih * tile_w + iw) = U4[i];
                        Vrow(i, 5, ic, in * tile_h * tile_w + ih * tile_w + iw) = U5[i];
                    }

                    Vrow(4, 0, ic, in * tile_h * tile_w + ih * tile_w + iw) = U04;
                    Vrow(4, 1, ic, in * tile_h * tile_w + ih * tile_w + iw) = U14;
                    Vrow(4, 2, ic, in * tile_h * tile_w + ih * tile_w + iw) = U24;
                    Vrow(4, 3, ic, in * tile_h * tile_w + ih * tile_w + iw) = U34;
                    Vrow(4, 4, ic, in * tile_h * tile_w + ih * tile_w + iw) = U44;
                    Vrow(4, 5, ic, in * tile_h * tile_w + ih * tile_w + iw) = U54;

                    Vrow(5, 0, ic, in * tile_h * tile_w + ih * tile_w + iw) = U05;
                    Vrow(5, 1, ic, in * tile_h * tile_w + ih * tile_w + iw) = U15;
                    Vrow(5, 2, ic, in * tile_h * tile_w + ih * tile_w + iw) = U25;
                    Vrow(5, 3, ic, in * tile_h * tile_w + ih * tile_w + iw) = U35;
                    Vrow(5, 4, ic, in * tile_h * tile_w + ih * tile_w + iw) = U45;
                    Vrow(5, 5, ic, in * tile_h * tile_w + ih * tile_w + iw) = U55;
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
#pragma omp parallel for collapse(2) private(in, ik, ih, iw, M0, M1, M2, M3, M4, M5, W0, W1, W04, W05, W14, W15, Z, hh, ww, i, j) if ((n * k) > 1)
    for (in = 0; in < n; in++)
        for (ik = 0; ik < k; ik++)
            for (ih = 0; ih < tile_h; ih++)
                for (iw = 0; iw < tile_w; iw++) {
                    // Z = (At @ M[..., ik, in * tile_h * tile_w + ih * tile_w + iw]) @ At.T
                    // Take advantage that because of the change in the previous block of nested loops, M is now contiguous in memory.
                    // Therefore, we are actually computing the following:
                    //     Z = (At @ M[in * tile_h * tile_w + ih * tile_w + iw, ik, ...]) @ At.T

                    // Load rows of M: 6x6
                    for (i = 0; i < 4; i++) {
                        M0[i] = Mrow(i, 0, ik, in * tile_h * tile_w + ih * tile_w + iw);
                        M1[i] = Mrow(i, 1, ik, in * tile_h * tile_w + ih * tile_w + iw);
                        M2[i] = Mrow(i, 2, ik, in * tile_h * tile_w + ih * tile_w + iw);
                        M3[i] = Mrow(i, 3, ik, in * tile_h * tile_w + ih * tile_w + iw);
                        M4[i] = Mrow(i, 4, ik, in * tile_h * tile_w + ih * tile_w + iw);
                        M5[i] = Mrow(i, 5, ik, in * tile_h * tile_w + ih * tile_w + iw);
                    }

                    // W_i  = A_row(i)  *  [ M0;M1;M2;M3 ] (rows of M), with
                    //    [      1,      1,      1,      1,      1,      0 ]     [ M00,  M01,  M02,  M03 | M04,  M05 ]     [ W00,  W01,  W02,  W03 | W04,  W05 ]
                    //    [      0,      1,     -1,      2,     -2,      1 ]  *  [ M10,  M11,  M12,  M13 | M14,  M15 ]  =  [ W10,  W11,  W12,  W13 | W14,  W15 ]
                    //                                                           [ M20,  M21,  M22,  M23 | M24,  M25 ]
                    //                                                           [ M30,  M31,  M32,  M33 | M34,  M35 ]
                    //                                                           [ M40,  M41,  M42,  M43 | M44,  M45 ]
                    //                                                           [ M50,  M51,  M52,  M53 | M54,  M55 ]
                    W0 = M0 + M1 + M2 + M3 + M4;
                    W04 = Mprow(0, 4) + Mprow(1, 4) + Mprow(2, 4) + Mprow(3, 4) + Mprow(4, 4);
                    W05 = Mprow(0, 5) + Mprow(1, 5) + Mprow(2, 5) + Mprow(3, 5) + Mprow(4, 5);

                    W1 = M1 - M2 + 2.0 * (M3 - M4) + M5;
                    W14 = Mprow(1, 4) - Mprow(2, 4) + 2.0 * (Mprow(3, 4) - Mprow(4, 4)) + Mprow(5, 4);
                    W15 = Mprow(1, 5) - Mprow(2, 5) + 2.0 * (Mprow(3, 5) - Mprow(4, 5)) + Mprow(5, 5);

                    // In contrast with cases 1) and 2), in this case we do not use vector instructions for this second gemm as
                    // the result is only 2x2 and we would be doing many innecessary flops
                    //    [      1,      1,      1,      1,      1,      0 ]     [ W00,  W10 ]     [ Z00,  Z01 ]
                    //    [      0,      1,     -1,      2,     -2,      1 ]  *  [ W01,  W11 ]  =  [ Z10,  Z11 ]
                    //                                                           [ W02,  W12 ]
                    //                                                           [ W03,  W13 ]
                    //                                                           [ W04,  W14 ]
                    //                                                           [ W05,  W15 ]
                    Z[0] = W0[0] + W0[1] + W0[2] + W0[3] + W04;
                    Z[2] = W1[0] + W1[1] + W1[2] + W1[3] + W14;

                    Z[1] = W0[1] - W0[2] + 2.0 * W0[3] - 2.0 * W04 + W05;
                    Z[3] = W1[1] - W1[2] + 2.0 * W1[3] - 2.0 * W14 + W15;

                    if (biases != NULL)
                        Z = Z + biases[ik];

                    if (bn == 'T')
                        Z = (((Z - running_mean[ik]) * inv_std[ik]) * gamma[ik]) + beta[ik];

                    if (relu == 'T')
                        Z = vmaxq_f32(Z, zeros);

                    hh = ih * s;
                    ww = iw * s;
                    // Yw[n, k, hh:hh+m, ww:ww+m] = Z[:min(m, H-hh), :min(m, W-ww)]
                    for (i = 0; i < min(m, ho - hh); i++)
                        for (j = 0; j < min(m, wo - ww); j++)
                            Yrow(in, ik, hh + i, ww + j) = Z[j * m + i];
                }
}

#ifdef TENSOR_FORMAT_NHWC
void conv_winograd_2x2_5x5_neon_fp32_nhwc
#else
void conv_winograd_2x2_5x5_neon_fp32_nchw
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
    conv_winograd_2x2_5x5_neon_fp32_nhwc_pre
#else
    conv_winograd_2x2_5x5_neon_fp32_nchw_pre
#endif
         (m, r, n, k, c, kh, kw, F, ldF1, ldF2, ldF3, U);

#ifdef TENSOR_FORMAT_NHWC
    conv_winograd_2x2_5x5_neon_fp32_nhwc_kernel
#else
    conv_winograd_2x2_5x5_neon_fp32_nchw_kernel
#endif
        (m, r, n, k, c, hi, wi, kh, kw, vpadding, hpadding,
         D, ldD1, ldD2, ldD3, Y, ldY1, ldY2, ldY3,
         biases, U, V, M, relu, bn, running_mean, inv_std,
         gamma, beta);
}
