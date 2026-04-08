# Vortex test suite

include(Catch)

# --- Index tests ---
add_executable(test_distance
    index/test_distance.cpp
)
target_link_libraries(test_distance PRIVATE vortex Catch2::Catch2WithMain)
target_include_directories(test_distance PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_distance)

add_executable(test_hnsw
    index/test_hnsw.cpp
)
target_link_libraries(test_hnsw PRIVATE vortex Catch2::Catch2WithMain)
target_include_directories(test_hnsw PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_hnsw)

# --- Pipeline tests ---
add_executable(test_pipeline
    pipeline/test_pipeline.cpp
)
target_link_libraries(test_pipeline PRIVATE vortex Catch2::Catch2WithMain)
target_include_directories(test_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_pipeline)

# --- Scheduler tests ---
add_executable(test_scheduler
    scheduler/test_scheduler.cpp
)
target_link_libraries(test_scheduler PRIVATE vortex Catch2::Catch2WithMain)
target_include_directories(test_scheduler PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_scheduler)

# --- Stress tests ---
add_executable(test_stress
    stress/test_concurrent_hnsw.cpp
)
target_link_libraries(test_stress PRIVATE vortex Catch2::Catch2WithMain)
target_include_directories(test_stress PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_stress)
