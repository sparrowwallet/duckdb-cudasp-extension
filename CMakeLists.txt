cmake_minimum_required(VERSION 3.18)

# Set extension name here
set(TARGET_NAME cudasp)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..
find_package(OpenSSL REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME} LANGUAGES CXX CUDA)
include_directories(src/include)
include_directories(gECC/include)

# Set CUDA architecture (adjust based on your GPU)
# Common values: 75 (Turing), 80 (Ampere), 86 (RTX 30xx), 89 (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89)

# Enable gECC optimizations
add_compile_definitions(GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS)
add_compile_definitions(PERSISTENT_L2_CACHE)
add_compile_definitions(OVERFLOW)
add_compile_definitions(SPPARK_32)

# Generate gECC constants (required for compilation)
set(GECC_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/gecc_generated")
file(MAKE_DIRECTORY ${GECC_GENERATED_DIR})

set(GECC_GENERATED_HEADERS
    "${GECC_GENERATED_DIR}/fp_constants.h"
    "${GECC_GENERATED_DIR}/fp_ops_cc_details.h"
    "${GECC_GENERATED_DIR}/ec_ops_add_details.h"
    "${GECC_GENERATED_DIR}/ec_ops_dbl_details.h"
    "${GECC_GENERATED_DIR}/fp_test_constants.h"
)

add_custom_command(
    OUTPUT ${GECC_GENERATED_HEADERS}
    COMMAND ${CMAKE_COMMAND} -E env python3
        "${CMAKE_CURRENT_SOURCE_DIR}/gECC/scripts/constants_generator.py"
        --out "${GECC_GENERATED_DIR}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/gECC/scripts/constants_generator.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/gECC/scripts/constants.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/gECC/scripts/ccgen.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/gECC/scripts/ec_ops.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/gECC/scripts/ec.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/gECC/scripts/field.py"
    COMMENT "Generating gECC constants..."
    VERBATIM
)

add_custom_target(gecc_constants DEPENDS ${GECC_GENERATED_HEADERS})

# Add generated headers to include path
include_directories(${GECC_GENERATED_DIR})

set(EXTENSION_SOURCES
    src/cudasp_extension.cu
    src/cudasp_gpu.cu
)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Ensure gECC constants are generated before building
add_dependencies(${EXTENSION_NAME} gecc_constants)
add_dependencies(${LOADABLE_EXTENSION_NAME} gecc_constants)

# Link OpenSSL and CUDA in both the static library and the loadable extension
target_link_libraries(${EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto CUDA::cudart)
target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto CUDA::cudart)

# Set CUDA-specific compile options
set_source_files_properties(src/cudasp_extension.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/cudasp_gpu.cu PROPERTIES LANGUAGE CUDA)
target_compile_options(${EXTENSION_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -diag-suppress=940  # Suppress false-positive "missing return statement" warnings from gECC library
>)
target_compile_options(${LOADABLE_EXTENSION_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -diag-suppress=940  # Suppress false-positive "missing return statement" warnings from gECC library
>)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
