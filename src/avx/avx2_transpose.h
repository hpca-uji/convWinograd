#pragma once

// Transpose 4x4 blocks within each lane
#define _MM_TRANSPOSE4_PS_M(M) \
	do { \
               _MM_TRANSPOSE4_PS(M[0], M[1], M[2], M[3]); \
	} while (0)

// http://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
#define _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7) \
	do { \
		__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7; \
		__m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7; \
		__t0 = _mm256_unpacklo_ps(row0, row1); \
		__t1 = _mm256_unpackhi_ps(row0, row1); \
		__t2 = _mm256_unpacklo_ps(row2, row3); \
		__t3 = _mm256_unpackhi_ps(row2, row3); \
		__t4 = _mm256_unpacklo_ps(row4, row5); \
		__t5 = _mm256_unpackhi_ps(row4, row5); \
		__t6 = _mm256_unpacklo_ps(row6, row7); \
		__t7 = _mm256_unpackhi_ps(row6, row7); \
		__tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0)); \
		__tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2)); \
		__tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0)); \
		__tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2)); \
		__tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0)); \
		__tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2)); \
		__tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0)); \
		__tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2)); \
		row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20); \
		row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20); \
		row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20); \
		row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20); \
		row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31); \
		row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31); \
		row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31); \
		row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31); \
	} while (0)

#define _MM_TRANSPOSE16_PS(row0, row1, row2, row3, row4, row5, row6, row7, \
                           row8, row9, rowa, rowb, rowc, rowd, rowe, rowf) \
        do { \
                __m512 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7, __t8, __t9, __ta, __tb, __tc, __td, __te, __tf; \
                __t0 = _mm512_unpacklo_ps(row0, row1); \
                __t1 = _mm512_unpackhi_ps(row0, row1); \
                __t2 = _mm512_unpacklo_ps(row2, row3); \
                __t3 = _mm512_unpackhi_ps(row2, row3); \
                __t4 = _mm512_unpacklo_ps(row4, row5); \
                __t5 = _mm512_unpackhi_ps(row4, row5); \
                __t6 = _mm512_unpacklo_ps(row6, row7); \
                __t7 = _mm512_unpackhi_ps(row6, row7); \
                __t8 = _mm512_unpacklo_ps(row8, row9); \
                __t9 = _mm512_unpackhi_ps(row8, row9); \
                __ta = _mm512_unpacklo_ps(rowa, rowb); \
                __tb = _mm512_unpackhi_ps(rowa, rowb); \
                __tc = _mm512_unpacklo_ps(rowc, rowd); \
                __td = _mm512_unpackhi_ps(rowc, rowd); \
                __te = _mm512_unpacklo_ps(rowe, rowf); \
                __tf = _mm512_unpackhi_ps(rowe, rowf); \
                row0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(__t0), _mm512_castps_pd(__t2))); \
                row1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(__t0), _mm512_castps_pd(__t2))); \
                row2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(__t1), _mm512_castps_pd(__t3))); \
                row3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(__t1), _mm512_castps_pd(__t3))); \
                row4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(__t4), _mm512_castps_pd(__t6))); \
                row5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(__t4), _mm512_castps_pd(__t6))); \
                row6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(__t5), _mm512_castps_pd(__t7))); \
                row7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(__t5), _mm512_castps_pd(__t7))); \
                row8 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(__t8), _mm512_castps_pd(__ta))); \
                row9 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(__t8), _mm512_castps_pd(__ta))); \
                rowa = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(__t9), _mm512_castps_pd(__tb))); \
                rowb = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(__t9), _mm512_castps_pd(__tb))); \
                rowc = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(__tc), _mm512_castps_pd(__te))); \
                rowd = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(__tc), _mm512_castps_pd(__te))); \
                rowe = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(__td), _mm512_castps_pd(__tf))); \
                rowf = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(__td), _mm512_castps_pd(__tf))); \
                __t0 = _mm512_shuffle_f32x4(row0, row4, 0x88); \
                __t1 = _mm512_shuffle_f32x4(row1, row5, 0x88); \
                __t2 = _mm512_shuffle_f32x4(row2, row6, 0x88); \
                __t3 = _mm512_shuffle_f32x4(row3, row7, 0x88); \
                __t4 = _mm512_shuffle_f32x4(row0, row4, 0xdd); \
                __t5 = _mm512_shuffle_f32x4(row1, row5, 0xdd); \
                __t6 = _mm512_shuffle_f32x4(row2, row6, 0xdd); \
                __t7 = _mm512_shuffle_f32x4(row3, row7, 0xdd); \
                __t8 = _mm512_shuffle_f32x4(row8, rowc, 0x88); \
                __t9 = _mm512_shuffle_f32x4(row9, rowd, 0x88); \
                __ta = _mm512_shuffle_f32x4(rowa, rowe, 0x88); \
                __tb = _mm512_shuffle_f32x4(rowb, rowf, 0x88); \
                __tc = _mm512_shuffle_f32x4(row8, rowc, 0xdd); \
                __td = _mm512_shuffle_f32x4(row9, rowd, 0xdd); \
                __te = _mm512_shuffle_f32x4(rowa, rowe, 0xdd); \
                __tf = _mm512_shuffle_f32x4(rowb, rowf, 0xdd); \
                row0 = _mm512_shuffle_f32x4(__t0, __t8, 0x88); \
                row1 = _mm512_shuffle_f32x4(__t1, __t9, 0x88); \
                row2 = _mm512_shuffle_f32x4(__t2, __ta, 0x88); \
                row3 = _mm512_shuffle_f32x4(__t3, __tb, 0x88); \
                row4 = _mm512_shuffle_f32x4(__t4, __tc, 0x88); \
                row5 = _mm512_shuffle_f32x4(__t5, __td, 0x88); \
                row6 = _mm512_shuffle_f32x4(__t6, __te, 0x88); \
                row7 = _mm512_shuffle_f32x4(__t7, __tf, 0x88); \
                row8 = _mm512_shuffle_f32x4(__t0, __t8, 0xdd); \
                row9 = _mm512_shuffle_f32x4(__t1, __t9, 0xdd); \
                rowa = _mm512_shuffle_f32x4(__t2, __ta, 0xdd); \
                rowb = _mm512_shuffle_f32x4(__t3, __tb, 0xdd); \
                rowc = _mm512_shuffle_f32x4(__t4, __tc, 0xdd); \
                rowd = _mm512_shuffle_f32x4(__t5, __td, 0xdd); \
                rowe = _mm512_shuffle_f32x4(__t6, __te, 0xdd); \
                rowf = _mm512_shuffle_f32x4(__t7, __tf, 0xdd); \
	} while (0)
