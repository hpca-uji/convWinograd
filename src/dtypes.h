#if defined(FP16)
  #define DTYPE _Float16
  #define gemm_microkernel_Cresident_neon_4x4_prefetch gemm_microkernel_Cresident_neon_4x4_prefetch_fp16
  #define gemm_microkernel_Cresident_neon_8x8_prefetch gemm_microkernel_Cresident_neon_8x8_prefetch_fp16
  #define gemm_microkernel_ABresident_neon_4x4         gemm_microkernel_ABresident_neon_4x4_prefetch_fp16
#elif defined(FP32)
  #define DTYPE float
  #define gemm_microkernel_Cresident_neon_4x4_prefetch gemm_microkernel_Cresident_neon_4x4_prefetch_fp32
  #define gemm_microkernel_Cresident_neon_8x8_prefetch gemm_microkernel_Cresident_neon_8x8_prefetch_fp32
  #define gemm_microkernel_ABresident_neon_4x4         gemm_microkernel_ABresident_neon_4x4_fp32
#elif defined(FP64)
  #define DTYPE double
  #define gemm_microkernel_Cresident_neon_4x4_prefetch gemm_microkernel_Cresident_neon_4x4_prefetch_fp64
  #define gemm_microkernel_Cresident_neon_8x8_prefetch gemm_microkernel_Cresident_neon_4x8_prefetch_fp64
  #define gemm_microkernel_ABresident_neon_4x4         gemm_microkernel_ABresident_neon_4x4_prefetch_fp64
#endif
