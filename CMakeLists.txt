cmake_minimum_required(VERSION 3.11 FATAL_ERROR)


project(libfluid VERSION 1.0.0 LANGUAGES CXX)


option(BUILD_SAMPLES "Build samples" ON)
option(BUILD_TESTS "Build tests" ON)

set(CMAKE_VERBOSE_MAKEFILE on)

find_package(mio REQUIRED)
find_package(glm REQUIRED)
find_package(OpenCL REQUIRED)
find_package(nlohmann_json REQUIRED)

find_package(OpenMP)
if (NOT TARGET OpenMP::OpenMP_CXX)
    # https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
            PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    set_property(TARGET OpenMP::OpenMP_CXX
            PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
endif ()

include_directories(third_party)
add_library(fluid
        include/fluid/zcurve.h
        include/fluid/cl_types.h
        include/fluid/clsph.hpp
        include/fluid/writer.hpp
        include/fluid/surface.hpp
        include/fluid/fluid.hpp
        include/fluid/cpusph.hpp
        include/fluid/clsph_type.h
        include/fluid/clsph_kernel.cl
        include/fluid/clmc_kernel.cl
        )

set_target_properties(fluid PROPERTIES LINKER_LANGUAGE CXX)
add_library(fluid::fluid ALIAS fluid)

target_include_directories(fluid
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        )

target_compile_features(fluid INTERFACE cxx_std_14)


target_link_libraries(fluid
        PUBLIC
        glm
        mio::mio
        OpenCL::OpenCL
        OpenMP::OpenMP_CXX
        nlohmann_json::nlohmann_json)

if (BUILD_SAMPLES)
    add_subdirectory(samples)
endif ()


if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()



