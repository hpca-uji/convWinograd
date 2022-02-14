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

inline void fvtrans_float32_4x4_neon_fp32( float32x4_t *A0, float32x4_t *A1, float32x4_t *A2, float32x4_t *A3 ) {
  float32x4x2_t V = vtrnq_f32 ( (float32x4_t) vtrn1q_f64 ( (float64x2_t) *A0, (float64x2_t) *A2 ),
                                (float32x4_t) vtrn1q_f64 ( (float64x2_t) *A1, (float64x2_t) *A3 ));
  float32x4x2_t W = vtrnq_f32 ( (float32x4_t) vtrn2q_f64 ( (float64x2_t) *A0, (float64x2_t) *A2 ),
                                (float32x4_t) vtrn2q_f64 ( (float64x2_t) *A1, (float64x2_t) *A3 ));
  *A0 = V.val[0];
  *A1 = V.val[1];
  *A2 = W.val[0];
  *A3 = W.val[1];
}
