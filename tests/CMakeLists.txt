cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(tests)

include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.7.0
)
FetchContent_MakeAvailable(Catch2)
#find_package(Catch2 REQUIRED)


add_executable(test-all test.cpp test-main.cpp)
target_link_libraries(test-all PRIVATE fluid::fluid)
target_link_libraries(test-all PRIVATE Catch2::Catch2)


#include(CTest)
#include(Catch)
#catch_discover_tests(run)


