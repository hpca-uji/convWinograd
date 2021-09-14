/* 
   Winograd

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
#include <sys/time.h>

#include "dtypes.h"

#define DTYPE float

#define Trow(a1,a2,a3,a4)  T[ (a1)*(ldT1)+(a2)*(ldT2)+(a3)*(ldT3)+(a4) ]
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]
#define Mrow(a1,a2)  M[ (a1)*(ldM)+(a2) ]

void generate_tensor4D( int m1, int m2, int m3, int m4, DTYPE *T, int ldT1, int ldT2, int ldT3 )
{
/*
 * Generate a 4D tensor with random entries
 * T      : Tensor
 *
 */
  int i1, i2, i3, i4;

  for ( i1=0; i1<m1; i1++ )
  for ( i2=0; i2<m2; i2++ )
  for ( i3=0; i3<m3; i3++ )
  for ( i4=0; i4<m4; i4++ )
#if defined(FP16)
        Trow(i1,i2,i3,i4) = ((((DTYPE) i1*m1+i2)/m1)/m2);
#else
        Trow(i1,i2,i3,i4) = ((DTYPE) rand())/RAND_MAX + 1.0;
#endif
}
/*===========================================================================*/
void print_tensor4D( char *name, int m1, int m2, int m3, int m4, DTYPE *T, int ldT1, int ldT2, int ldT3 )
{
/*
 * Print a 4D tensor to standard output
 * name   : Label for vector name
 * m      : Dimension
 * v      : Vector
 *
 */
  int i1, i2, i3, i4;

  for ( i1=0; i1<m1; i1++ )
  for ( i2=0; i2<m2; i2++ )
  for ( i3=0; i3<m3; i3++ )
  for ( i4=0; i4<m4; i4++ )
#if defined(FP16)
        printf( "%s[%d,%d,%d,%d] = %8.2e;\n", name, i1, i2, i3, i4, ((double) Trow(i1, i2, i3, i4)) );
#elif defined(FP32)
        printf( "%s[%d,%d,%d,%d] = %14.8e;\n", name, i1, i2, i3, i4, ((double) Trow(i1, i2, i3, i4)) );
#elif defined(FP64)
        printf( "%s[%d,%d,%d,%d] = %22.16e;\n", name, i1, i2, i3, i4, ((double) Trow(i1, i2, i3, i4)) );
#endif
}

/*===========================================================================*/
double dclock()
{
/* 
 * Timer
 *
 */
  struct timeval  tv;
  // struct timezone tz;

  gettimeofday( &tv, NULL );   

  return (double) (tv.tv_sec + tv.tv_usec*1.0e-6);
}
/*===========================================================================*/
void print_matrix( char *name, char orderM, int m, int n, DTYPE *M, int ldM )
{
/*
 * Print a matrix to standard output
 * name   : Label for matrix name
 * m      : Row dimension
 * n      : Column dimension
 * A      : Matrix
 *
 */
  int i, j;
  
  if ( orderM=='C' )
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ )
#if defined(FP16)
        printf( "%s[%d,%d] = %8.2e;\n", name, i, j, ((double) Mcol(i,j)) );
#elif defined(FP32)
        printf( "%s[%d,%d] = %14.8e;\n", name, i, j, ((double) Mcol(i,j)) );
#elif defined(FP64)
        printf( "%s[%d,%d] = %22.16e;\n", name, i, j, ((double) Mcol(i,j)) );
#endif
  else
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ )
#if defined(FP16)
        printf( "%s[%d,%d] = %8.2e;\n", name, i, j, ((double) Mrow(i,j)) );
#elif defined(FP32)
        printf( "%s[%d,%d] = %14.8e;\n", name, i, j, ((double) Mrow(i,j)) );
#elif defined(FP64)
        printf( "%s[%d,%d] = %22.16e;\n", name, i, j, ((double) Mrow(i,j)) );
#endif
}

