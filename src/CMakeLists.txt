# src CMakeLists.txt


# ========================================================================
# Macro called from src subdirectories to set the library sources
# ------------------------------------------------------------------------
macro(set_parent_LIBSOURCES SOURCES)
    foreach (SOURCE IN LISTS SOURCES)
        # Add NCHW object (not an actual library)
        add_library(${SOURCE}_nchw OBJECT ${SOURCE}.c)
        target_compile_options(${SOURCE}_nchw PRIVATE -DTENSOR_FORMAT_NCHW)
        list(APPEND LIBSOURCES $<TARGET_OBJECTS:${SOURCE}_nchw>)
        # Add NHWc object (not an actual library)
        add_library(${SOURCE}_nhwc OBJECT ${SOURCE}.c)
        target_compile_options(${SOURCE}_nhwc PRIVATE -DTENSOR_FORMAT_NHWC)
        list(APPEND LIBSOURCES $<TARGET_OBJECTS:${SOURCE}_nhwc>)
    endforeach ()
    set(LIBSOURCES "${LIBSOURCES}" PARENT_SCOPE)
endmacro()
# ========================================================================


# ========================================================================
# convwinograd library compilation, linking and installation
# ------------------------------------------------------------------------
add_library(convwinograd SHARED)

# Common sources to all architectures
add_subdirectory(common)
target_sources(convwinograd PUBLIC ${LIBSOURCES})

# Architecture dependant sources
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    find_package(AVX QUIET)
    if (C_AVX_FOUND)
        add_compile_options(-DC_AVX_FOUND ${C_AVX_FLAGS})
    endif ()
    add_subdirectory(sse)
    add_subdirectory(avx)
    target_sources(convwinograd PUBLIC ${LIBSOURCES})
    #
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    add_subdirectory(neon)
    target_sources(convwinograd PUBLIC ${LIBSOURCES})
    #
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "riscv")
    add_subdirectory(riscv)
    target_sources(convwinograd PUBLIC ${LIBSOURCES})
    #
else ()
    message(SEND_ERROR "Architecture '${CMAKE_SYSTEM_PROCESSOR}' is not yet supported!")
endif ()

# Add links to the OpenMP and BLAS (or gemm.c) libraries and propagate these to dependant targets (PUBLIC option)
target_link_libraries(convwinograd PUBLIC OpenMP::OpenMP_C)
if (BLAS_FOUND)
    target_link_libraries(convwinograd PUBLIC BLAS::BLAS)
else ()
    target_sources(convwinograd PUBLIC gemm.c)
endif ()

# Add compile options for convwinograd library and propagate them to dependant targets (PUBLIC option)
target_compile_options(convwinograd PUBLIC
        -Wall
        )

# Install rule
install(TARGETS convwinograd DESTINATION lib)
# ========================================================================
