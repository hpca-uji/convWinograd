# convwinograd CMakeLists.txt

cmake_minimum_required(VERSION 3.13)

project(ConvWinograd VERSION 1.0 LANGUAGES C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Add $ENV{LD_LIBRARY_PATH} to CMAKE_PREFIX_PATH
if (DEFINED ENV{LD_LIBRARY_PATH})
    string(REPLACE ":" ";" LD_LIBRARY_PATH_AS_LIST $ENV{LD_LIBRARY_PATH})
    list(APPEND CMAKE_PREFIX_PATH "${LD_LIBRARY_PATH_AS_LIST}")
endif ()

find_package(OpenMP REQUIRED)

# ========================================================================
# Find BLAS and record which version has been found
# ------------------------------------------------------------------------
set(SUPPORTED_VENDORS FLAME;Intel10_64ilp)
set(CBLAS_TYPES CBLAS;CBLAS)
if (BLA_VENDOR)
    if (BLA_VENDOR STREQUAL "None")
        set(CBLAS_TYPE "OURS")
    else ()
        foreach (VENDOR TYPE IN ZIP_LISTS SUPPORTED_VENDORS CBLAS_TYPES)
            if (BLA_VENDOR STREQUAL ${VENDOR})
                message(STATUS "Searching CBLAS for BLA_VENDOR=${BLA_VENDOR}...")
                set(CBLAS_TYPE ${TYPE})
                find_package(BLAS)
                continue()
            endif ()
        endforeach ()
        if (NOT CBLAS_TYPE)
            message(SEND_ERROR "BLA_VENDOR '${BLA_VENDOR}' is not supported (see README.md)!")
        else ()
            if (BLAS_FOUND)
                message(STATUS "...${CBLAS_TYPE} (${BLA_VENDOR}) found!")
            else ()
                message(SEND_ERROR "CBLAS for BLA_VENDOR=${BLA_VENDOR} not found!")
            endif ()
        endif ()
    endif ()
else ()
    message(STATUS "BLA_VENDOR not specified, search supported vendors...")
    foreach (VENDOR TYPE IN ZIP_LISTS SUPPORTED_VENDORS CBLAS_TYPES)
        set(BLA_VENDOR ${VENDOR})
        set(CBLAS_TYPE ${TYPE})
        find_package(BLAS)
        if (BLAS_FOUND)
            message(STATUS "...${CBLAS_TYPE} (${BLA_VENDOR}) found!")
            break()
        endif ()
    endforeach ()
    if (NOT BLAS_FOUND)
        message(SEND_ERROR "CBLAS not found! Set -D BLA_VENDOR=None to force the bundled one.")
    endif ()
endif ()
if (BLAS_FOUND)
    add_compile_options(-DCBLAS_TYPE_${CBLAS_TYPE} -DBLA_VENDOR_${BLA_VENDOR})
    if (BLAS_LIBRARIES)
        list(GET BLAS_LIBRARIES 0 FIRST_BLAS_LIBRARY)
        get_filename_component(FIRST_BLAS_LIBRARY_PATH ${FIRST_BLAS_LIBRARY} DIRECTORY)
        include_directories(SYSTEM PUBLIC
                "${FIRST_BLAS_LIBRARY_PATH}/../include"
                "${FIRST_BLAS_LIBRARY_PATH}/../include/blis"
                )
    endif ()
endif ()
# ========================================================================


# ========================================================================
# Common compilation options and include directories
# ------------------------------------------------------------------------
# Common compilation options to all objects
add_compile_options(
        -O3 -mtune=native -march=native
        -ftree-vectorize
        -fopenmp
        -fPIC
)
# Include SYSTEM directories (no compilation warnings due to these)
include_directories(
        SYSTEM PUBLIC
        "${CMAKE_INSTALL_PREFIX}/include"
)
if (CMAKE_PREFIX_PATH)
    include_directories(
            SYSTEM PUBLIC
            "${CMAKE_PREFIX_PATH}/include"
            "${CMAKE_PREFIX_PATH}"
    )
endif ()
# ========================================================================


add_subdirectory(src)

option(COMPILE_TESTS "Compile tests" OFF)

if (COMPILE_TESTS)
    add_subdirectory(tests)
endif ()
