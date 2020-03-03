libfluid
========

Generic fluid simulation solver based on SPH method as described in [Position Based Fluids(ACM TOG 32)](http://mmacklin.com/pbf_sig_preprint.pdf).

![](a.gif)

![](b.gif)

## Build dependencies

Build dependencies:

 * CMake >= 3.11
 * GGG/Clang/MSVC

Library dependencies:

 * OpenMP >= 2.0 (2.0 is supported but with degraded proformance)
 * OpenCL (runtime must support OpenCL 1.2 or newer)
 * Catch2
 * GLM
 * mio
 * nlohmann-json
 
If you are using vcpkg, install the following dependencies:

    vcpkg install opencl glm mio nlohmann-json catch2

On MacOS, first install OpenMP:

    brew install libomp    

# Compiling

On Windows(PowerShell):

    cmake -Bbuild -H\. -DCMAKE_TOOLCHAIN_FILE=C:\Users\<user>\vcpkg\scripts\buildsystems\vcpkg.cmake -DCMAKE_BUILD_TYPE=Release

On Linux:

    cmake -Bbuild -H. -DCMAKE_TOOLCHAIN_FILE=/home/<user>/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release

On MacOS:

    cmake -Bbuild -H. -DCMAKE_TOOLCHAIN_FILE=/User/<user>/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release

Then compile:

    cmake --build build --target main --config Release



To compile with Intel C++ Compiler(icc):

First install Intel Parallel Studio XE 2018/2019, be sure to select the `Intel C++ Compiler` option.

On Linux:

    export CC=icc
    export CXX=icc
    
Then run the usual cmake commands.


On Windows(tested to work on XE 2018 only):

First, ensure vcpkg has required x64 packages:

    vcpkg install <required packages...> --triplet x64-windows

Then:

    cmake -Bbuild -H\. -DCMAKE_TOOLCHAIN_FILE=C:\Users\<user>\vcpkg\scripts\buildsystems\vcpkg.cmake -DCMAKE_BUILD_TYPE=Release -G"Visual Studio 15 2017 Win64"  -T"Intel C++ Compiler 18.0"
    
    cmake --build build --target main --config Release
    

## Licence

    Copyright 2019 WEI CHEN LIN
    
    Licensed under the Apache License, Version 2.0 (the "License");e "License");
    you may not use this file except in compliance with the License. the License.
    You may obtain a copy of the License at
    
       http://www.apache.org/licenses/LICENSE-2.0writing, software
    "AS IS" BASIS,
    Unless required by applicable law or agreed to in writing, softwarer express or implied.
    distributed under the License is distributed on an "AS IS" BASIS, permissions and
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.