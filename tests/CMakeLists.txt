project(tests)

include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.7.0
)
#FetchContent_MakeAvailable(Catch2)
#find_package(Catch2 REQUIRED)

FetchContent_GetProperties(catch2)
if(NOT catch2_POPULATED)
    FetchContent_Populate(catch2)
    add_subdirectory(${catch2_SOURCE_DIR} ${catch2_BINARY_DIR})
endif()


add_executable(test-all test.cpp test-main.cpp)
target_link_libraries(test-all PRIVATE fluid::fluid)
target_link_libraries(test-all PRIVATE Catch2::Catch2)


#include(CTest)
#include(Catch)
#catch_discover_tests(run)


