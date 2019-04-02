cmake_minimum_required(VERSION 3.11 FATAL_ERROR)


project(libfluid VERSION 1.0.0)


option(BUILD_SAMPLES "Build samples" ON)
option(BUILD_TESTS "Build tests" ON)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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


add_library(fluid INTERFACE)

include_directories(third_party)
target_include_directories(fluid INTERFACE
        include/fluid
        include/fluid/oclsph_kernel.h
        third_party
        )

#set_source_files_properties(
#        clmc_kernel.cl
#        clsph_kernel.cl
#        PROPERTIES LANGUAGE CXX)

add_library(fluid::fluid ALIAS fluid)

target_include_directories(fluid
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>

        )

target_compile_features(fluid INTERFACE cxx_std_14)


target_link_libraries(fluid
        INTERFACE
        nlohmann_json::nlohmann_json
        glm
        mio::mio
        OpenCL::OpenCL
        OpenMP::OpenMP_CXX
        )


if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # CL2 typedefs cl types with attributes
    target_compile_options(fluid INTERFACE -Wno-ignored-attributes)
endif ()

if (BUILD_SAMPLES)
    add_subdirectory(samples)
endif ()


if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()



