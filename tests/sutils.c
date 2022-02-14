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
#include <time.h>

#include "../src/dtypes.h"

#define DTYPE float

#define Trow(a1, a2, a3, a4)  T[ (a1)*(ldT1)+(a2)*(ldT2)+(a3)*(ldT3)+(a4) ]
#define Mcol(a1, a2)  M[ (a2)*(ldM)+(a1) ]
#define Mrow(a1, a2)  M[ (a1)*(ldM)+(a2) ]

/**
 * Generates a 4D tensor with random entries
 *
 * @param m1
 * @param m2
 * @param m3
 * @param m4
 * @param T The 4D tensor
 * @param ldT1
 * @param ldT2
 * @param ldT3
 */
void generate_tensor4D(int m1, int m2, int m3, int m4, DTYPE *T, int ldT1, int ldT2, int ldT3) {
    int i1, i2, i3, i4;

    for (i1 = 0; i1 < m1; i1++)
        for (i2 = 0; i2 < m2; i2++)
            for (i3 = 0; i3 < m3; i3++)
                for (i4 = 0; i4 < m4; i4++) {
#if defined(FP16)
                    Trow(i1, i2, i3, i4) = (((DTYPE) i1 * m1 + i2) / m1) / m2;
#else
                    Trow(i1, i2, i3, i4) = ((DTYPE) rand()) / ((DTYPE) RAND_MAX) + 1.0;
#endif
                }
}

/**
 * Prints the given 4D tensor to the standard output
 *
 * @param name Label for the 4D tensor
 * @param m1
 * @param m2
 * @param m3
 * @param m4
 * @param T The 4D tensor
 * @param ldT1
 * @param ldT2
 * @param ldT3
 */
void print_tensor4D(char *name, int m1, int m2, int m3, int m4, DTYPE *T, int ldT1, int ldT2, int ldT3) {
    int i1, i2, i3, i4;

    for (i1 = 0; i1 < m1; i1++)
        for (i2 = 0; i2 < m2; i2++)
            for (i3 = 0; i3 < m3; i3++)
                for (i4 = 0; i4 < m4; i4++) {
#if defined(FP16)
                    printf("%s[%d,%d,%d,%d] = %8.2e;\n", name, i1, i2, i3, i4, ((double) Trow(i1, i2, i3, i4)));
#elif defined(FP32)
                    printf("%s[%d,%d,%d,%d] = %14.8e;\n", name, i1, i2, i3, i4, ((double) Trow(i1, i2, i3, i4)));
#elif defined(FP64)
                    printf("%s[%d,%d,%d,%d] = %22.16e;\n", name, i1, i2, i3, i4, ((double) Trow(i1, i2, i3, i4)));
#endif
                }
}

/**
 * Provides the number of seconds and nanoseconds from a point in the past as a double
 *
 * @return The number of seconds and nanoseconds from a point in the past
 */
double dclock() {
    /*
     * From man gettimeofday:
     *
     *  The  time returned by gettimeofday() is affected by discontinuous jumps in the system time (e.g., if the system
     *  administrator manually changes the system time).  If you need a monotonically increasing clock,
     *  see clock_gettime(2).
     *
     */
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (double) tp.tv_sec + (double) tp.tv_nsec * 1.0e-9;
}

/**
 * Prints a matrix to the standard output
 *
 * @param name Label for the matrix
 * @param orderM
 * @param m Number of rows
 * @param n Number of columns
 * @param M The matrix
 * @param ldM
 */
void print_matrix(char *name, char orderM, int m, int n, DTYPE *M, int ldM) {
    int i, j;

    if (orderM == 'C')
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++) {
#if defined(FP16)
                printf("%s[%d,%d] = %8.2e;\n", name, i, j, ((double) Mcol(i, j)));
#elif defined(FP323)
                printf("%s[%d,%d] = %14.8e;\n", name, i, j, ((double) Mcol(i, j)));
#elif defined(FP64)
                printf("%s[%d,%d] = %22.16e;\n", name, i, j, ((double) Mcol(i, j)));
#endif
            }
    else
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++) {
#if defined(FP16)
                printf("%s[%d,%d] = %8.2e;\n", name, i, j, ((double) Mrow(i, j)));
#elif defined(FP32)
                printf("%s[%d,%d] = %14.8e;\n", name, i, j, ((double) Mrow(i, j)));
#elif defined(FP64)
                printf("%s[%d,%d] = %22.16e;\n", name, i, j, ((double) Mrow(i, j)));
#endif
            }
}

