cmake_minimum_required(VERSION 3.0)
project("Parallel Horner" VERSION 1.0.0)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

FIND_PACKAGE(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

FIND_PACKAGE(GMP REQUIRED)
FIND_PACKAGE(PkgConfig)

# Benchmark
SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "no benchmark testing" FORCE)
SET(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "no benchmark testing" FORCE)
SET(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "no benchmark install" FORCE)
add_subdirectory(benchmark)
#include_directories(benchmark/include)

SET(INSTALL_GTEST OFF CACHE BOOL "no gtest install" FORCE)
SET(BUILD_GMOCK ON CACHE BOOL "build gmock" FORCE)
add_subdirectory(googletest)

SET(BUILD_TESTING OFF CACHE BOOL "no glog tests" FORCE)
add_subdirectory(glog)

add_executable(main main.cpp)
target_link_libraries(main benchmark ${GMPXX_LIBRARIES} ${GMP_LIBRARIES} glog
    gtest_main)

include(FindASan)

# Unit Tests
enable_testing()
add_executable(pHorner_tests EXCLUDE_FROM_ALL pHorner_tests.cpp)
target_link_libraries(pHorner_tests
    ${GMP_LIBRARIES} ${GMPXX_LIBRARIES} gtest_main gmock_main glog)
add_test(Tests pHorner_tests)
add_custom_target(tests DEPENDS pHorner_tests)
add_custom_target(run-tests COMMAND ${CMAKE_CTEST_COMMAND}
                            DEPENDS pHorner_tests)
