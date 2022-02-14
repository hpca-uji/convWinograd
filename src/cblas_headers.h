// Include CBLAS headers depending on CBLAS_TYPE

#if BLA_VENDOR_Intel10_64ilp

#if __has_include("mkl/mkl.h")

#include <mkl/mkl.h>

#else

#include <mkl.h>

#endif

#elif CBLAS_TYPE_CBLAS

#include <cblas.h>

#elif CBLAS_TYPE_OURS

#include "gemm.h"

#else

#warning "BLAS headers have not been loaded!"

#endif
