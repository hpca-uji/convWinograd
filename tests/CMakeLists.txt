# tests/CMakeLists.txt

# ========================================================================
# Add data type compilation option (force FP32 if no one is specified)
# ------------------------------------------------------------------------
if (FP16)
    add_compile_options(-DFP16)
elseif (FP64)
    add_compile_options(-DFP64)
else ()
    add_compile_options(-DFP32)
endif ()
# ========================================================================

add_executable(test_winograd test_winograd.c sutils.c)
target_link_libraries(test_winograd PRIVATE convwinograd m)

add_custom_target(
        run_test_winograd
        COMMAND ${PROJECT_SOURCE_DIR}/tests/test_winograd.sh
        DEPENDS test_winograd
)
