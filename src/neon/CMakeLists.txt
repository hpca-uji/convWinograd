# src/neon CMakeLists.txt

# GLOB could be used to gather all the sources, but it is not recommended, as when a source is added or removed, the
# cmake generator could not reliably detect it.

set(SOURCES
        conv_winograd_2x2_3x3_neon_fp32;
        conv_winograd_2x2_5x5_neon_fp32;
        conv_winograd_3x3_2x2_neon_fp32;
        conv_winograd_4x4_3x3_neon_fp32;
        )

set_parent_LIBSOURCES("${SOURCES}")
