cmake_minimum_required(VERSION 3.7 FATAL_ERROR)


if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using vcpkg:  $ENV{VCPKG_ROOT}")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif ()


project(libfluid)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)




#set(MSGPACK_ENABLE_STATIC true)
#set(MSGPACK_ENABLE_SHARED false)

find_package(Catch2 REQUIRED)
find_package(glm REQUIRED)
find_package(mio REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenMP  REQUIRED)
find_package(nlohmann_json REQUIRED)

if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
-Wall -pedantic \
-fno-omit-frame-pointer -fsanitize=address \
-O2 -march=native \
-fopenmp \
-g")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -lasan")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -pedantic \
-Ofast -funsafe-math-optimizations -march=native \
-fopenmp \
-g")

#find_package(BoostCompute CONFIG REQUIRED)

include_directories(${BoostCompute_INCLUDE_DIRS})

#add_library(libfluid Octree.hpp library.cpp library.hpp main.cpp library.cpp)

add_executable(main cl.hpp nanoflann.hpp Octree.hpp surface.hpp fluid.hpp main.cpp)
add_executable(libfluid test.cpp test-main.cpp)

target_link_libraries(libfluid PRIVATE Catch2::Catch2)
target_link_libraries(libfluid PRIVATE glm)
target_link_libraries(libfluid PRIVATE mio::mio)
target_link_libraries(libfluid PRIVATE OpenCL::OpenCL)

target_link_libraries(main PRIVATE Catch2::Catch2)
target_link_libraries(main PRIVATE glm)
target_link_libraries(main PRIVATE mio::mio)
target_link_libraries(main PRIVATE OpenCL::OpenCL)
#target_link_libraries(main PRIVATE stdc++fs)
target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)


include(CTest)
include(Catch)
catch_discover_tests(libfluid)