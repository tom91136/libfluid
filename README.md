libfluid
========

Generic fluid simulation solver based on SPH method as described in [Position Based Fluids(ACM TOG 32)](http://mmacklin.com/pbf_sig_preprint.pdf)

## Build dependencies

Build dependencies:

 * cmake
 * gcc/clang/msvc

Library dependencies:

 * boost-compute
 * OpenMP
 * OpenCL
 * Catch2
 * GLM
 * mio
 
If you are using vcpkg, install the following dependencies:

    vcpkg install boost glm mio opencl nlohmann-json catch2


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