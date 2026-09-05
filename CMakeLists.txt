cmake_minimum_required(VERSION 3.16)
project(sparse0 C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

# static llama.cpp linked into a shared object needs PIC
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# portable x86-64-v3 binary: no -march=native, no AVX-512, no AMX
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_AVX ON CACHE BOOL "" FORCE)
  set(GGML_AVX2 ON CACHE BOOL "" FORCE)
  set(GGML_FMA ON CACHE BOOL "" FORCE)
  set(GGML_F16C ON CACHE BOOL "" FORCE)
  set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
  set(GGML_AVX512_VBMI OFF CACHE BOOL "" FORCE)
  set(GGML_AVX512_VNNI OFF CACHE BOOL "" FORCE)
  set(GGML_AVX512_BF16 OFF CACHE BOOL "" FORCE)
  set(GGML_AMX_TILE OFF CACHE BOOL "" FORCE)
  set(GGML_AMX_INT8 OFF CACHE BOOL "" FORCE)
  set(GGML_AMX_BF16 OFF CACHE BOOL "" FORCE)
endif()
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(llama
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
  GIT_TAG        5d5cb4c3a4ea8769490d39a275ee49a45184774d
  GIT_SHALLOW    OFF)
FetchContent_MakeAvailable(llama)

set(UTF8PROC_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(utf8proc
  GIT_REPOSITORY https://github.com/JuliaStrings/utf8proc
  GIT_TAG        e5e799221b45bbb90f5fdc5c69b6b8dfbf017e78
  GIT_SHALLOW    OFF)
FetchContent_MakeAvailable(utf8proc)

add_library(sparse0 SHARED src/sparse0.c)
set_target_properties(sparse0 PROPERTIES PREFIX "" OUTPUT_NAME "sparse0")
target_include_directories(sparse0 PRIVATE
    ${SQLITE_INCLUDE_DIR}
    ${llama_SOURCE_DIR}/include
    ${llama_SOURCE_DIR}/ggml/include)
target_link_libraries(sparse0 PRIVATE llama utf8proc m)
