cmake_minimum_required(VERSION 3.20)
project(fleetwatch_aggregator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(fwagg
  src/iou.cpp
  src/match.cpp
  src/prcurve.cpp
  src/map.cpp
  src/json.cpp
  src/binproto.cpp
)
target_include_directories(fwagg PUBLIC src)

add_executable(fleetwatch-aggregator src/main.cpp)
target_link_libraries(fleetwatch-aggregator PRIVATE fwagg)

# ---- Tests ----
option(FWAGG_BUILD_TESTS "Build aggregator tests" ON)
if(FWAGG_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(fwagg_tests
    tests/test_iou.cpp
    tests/test_match.cpp
    tests/test_prcurve.cpp
    tests/test_map.cpp
    tests/test_json.cpp
    tests/test_binproto.cpp
  )
  target_link_libraries(fwagg_tests PRIVATE fwagg GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(fwagg_tests)
endif()
