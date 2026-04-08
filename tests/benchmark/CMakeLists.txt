# Benchmark executables

add_executable(bench_hnsw bench_hnsw.cpp)
target_link_libraries(bench_hnsw PRIVATE vortex)
target_include_directories(bench_hnsw PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(bench_pipeline bench_pipeline.cpp)
target_link_libraries(bench_pipeline PRIVATE vortex)
target_include_directories(bench_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/src)
