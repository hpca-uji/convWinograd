# src/common CMakeLists.txt

# GLOB could be used to gather all the sources, but it is not recommended, as when a source is added or removed, the
# cmake generator could not reliably detect it.

set(SOURCES
        conv_winograd_fp32;
        conv_winograd_2x2_3x3_native_fp32;
        conv_winograd_workspace_alloc;
        )

set_parent_LIBSOURCES("${SOURCES}")
