project(samples)


include(ExternalProject)


ExternalProject_Add(
        bunny
        PREFIX "demo"
        URL http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz
        URL_HASH SHA512=59e7b43db838dbe6f02fda5e844d18e190d32d2ca1a83dc9f6b1aaed43e0340fc1e8ecabed6fffdac9f85bd34e7e93b5d8a17283d59ea3c14977a0372785d2bd
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE fluid::fluid)

if (MSVC)
    target_compile_options(main
            PUBLIC
            /Wp64
            /W3
            /Zc:wchar_t
            )
    set(RELEASE_OPTIONS /MT /GF /Gy /O2 /arch:AVX2 /fp:fast)
    set(DEBUG_OPTIONS /MTd)
else ()
    target_compile_options(main
            PUBLIC
            -Wall
            -Wextra
            #            -Wshadow
            #            -Wold-style-cast
            -Wcast-align
            -Wpedantic
            -Werror=return-type
            )
    set(RELEASE_OPTIONS -Ofast -funsafe-math-optimizations -march=native -g)
    set(DEBUG_OPTIONS  -fno-omit-frame-pointer -fsanitize=address -march=native)
    target_link_libraries(main PUBLIC $<$<CONFIG:DEBUG>:-Wl,-lasan>)

endif ()

target_compile_options(main PUBLIC "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")
target_compile_options(main PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")


#install(TARGETS ${PROJECT_NAME} DESTINATION examples)
