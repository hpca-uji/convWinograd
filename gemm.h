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
#include <math.h>
#include <string.h>

#include "dtypes.h"

void gemm( char orderA, char orderB, char orderC,
           char transA, char transB,
           int m, int n, int k,
           DTYPE alpha, DTYPE *A, int ldA,
                        DTYPE *B, int ldB,
           DTYPE beta,  DTYPE *C, int ldC );
