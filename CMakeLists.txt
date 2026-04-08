cmake_minimum_required(VERSION 3.20)
project(vortex VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# --- Options ---
option(VORTEX_ENABLE_TSAN "Enable Thread Sanitizer" OFF)
if(VORTEX_ENABLE_TSAN)
    add_compile_options(-fsanitize=thread -g)
    add_link_options(-fsanitize=thread)
endif()

option(VORTEX_ENABLE_ASAN "Enable Address Sanitizer" OFF)
if(VORTEX_ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
endif()

option(VORTEX_BUILD_E2E "Build E2E Forge+Vortex rag_agent" OFF)
option(VORTEX_BUILD_BENCHMARKS "Build benchmark suite" OFF)

# --- Dependencies ---
include(FetchContent)

FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3 GIT_SHALLOW TRUE)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0 GIT_SHALLOW TRUE)

FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.15.3 GIT_SHALLOW TRUE)

FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2 GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(nlohmann_json spdlog httplib Catch2)

# --- Forge core (submodule) ---
# Forge's core primitives: thread pool, MPSC queue, future/promise, etc.
# We compile thread_pool.cpp and link the headers.
set(FORGE_DIR ${CMAKE_SOURCE_DIR}/forge)

add_library(forge-core STATIC
    ${FORGE_DIR}/src/core/thread_pool.cpp
)
target_include_directories(forge-core PUBLIC ${FORGE_DIR}/src)
target_link_libraries(forge-core PUBLIC spdlog::spdlog)

# --- Vortex library ---
add_library(vortex STATIC
    src/index/hnsw_index.cpp
    src/index/distance.cpp
    src/pipeline/rag_pipeline.cpp
    src/pipeline/api_embedder.cpp
    src/pipeline/document_processor.cpp
    src/scheduler/batch_scheduler.cpp
    src/scheduler/prefix_cache.cpp
    src/server/http_server.cpp
)
target_include_directories(vortex PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(vortex PUBLIC
    forge-core
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    httplib::httplib
)

# --- Vortex standalone server ---
add_executable(vortex-server src/main.cpp)
target_link_libraries(vortex-server PRIVATE vortex)

# --- Tests ---
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Benchmarks ---
if(VORTEX_BUILD_BENCHMARKS)
    add_subdirectory(tests/benchmark)
endif()

# --- E2E: Forge agent + Vortex retrieval ---
if(VORTEX_BUILD_E2E)
    add_subdirectory(e2e/rag_agent)
endif()
