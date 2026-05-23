cmake_minimum_required(VERSION 3.20)
project(LibtorchOpenCVTutorials)


set(CMAKE_CXX_STANDARD 20)
set(TORCH_CUDA_ARCH_LIST "7.5")
set(CAFFE2_USE_CUDNN 9)
set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/x64-linux/share/opencv4")
add_subdirectory(external/csv-parser)
find_package(OpenCV CONFIG REQUIRED)
find_package(DataFrame REQUIRED)
add_executable(LibtorchOpenCVTutorials main.cpp
        external/third_party/doctest.cpp
        external/third_party/doctest.hpp
        src/BasicFunctionality/readingImage.cpp
        src/BasicFunctionality/BasicOperationOnTensors.cpp
        src/autogradsInLibtorch/autogradsincpp.cpp
        external/smartStore/lib/tinyxml2/tinyxml2.cpp
        external/smartStore/src/versionForMigration/MigrationRegistry.cpp
        src/ReadingAndWriting/readingCsv.cpp
        src/ReadingAndWriting/readingAndWritingPrevalidation.cpp
        src/TrainingPipeline/TrainingPipeline.cpp
        src/TrainingPipeline/TrainingNNModule.cpp
        src/TrainingPipeline/TrainingNNDataLoader.cpp
        src/ANN/ann_fashion_mnist.cpp
        src/ANNGPUBASED/DataProcessing.hpp
        src/ANNGPUBASED/DataProcessing.cpp
        src/ANNGPUBASED/CustomDatasets.hpp
        src/ANNGPUBASED/CustomDatasets.cpp
        src/ANNGPUBASED/CNNImple.hpp
        src/ANNGPUBASED/CNNImple.cpp
        src/ANNGPUBASED/ANNGPUBASED.cpp
        src/TensorManipulation/tensor.cpp
        src/LinearRegression/LinearRegression.cpp
)

target_link_libraries(LibtorchOpenCVTutorials PRIVATE)

target_include_directories(LibtorchOpenCVTutorials PRIVATE
        "${CMAKE_SOURCE_DIR}/external/libtorch/include"
        "${CMAKE_SOURCE_DIR}/external/libtorch/include/torch/csrc/api/include"
        "${OpenCV_INCLUDE_DIRS}"
        ${PROJECT_SOURCE_DIR}/external/smartStore/src
        ${PROJECT_SOURCE_DIR}/external/smartStore/lib/tinyxml2
        ${PROJECT_SOURCE_DIR}/external/smartStore/lib
        ${PROJECT_SOURCE_DIR}/external/third_party
)
target_link_libraries(LibtorchOpenCVTutorials PRIVATE DataFrame::DataFrame)
target_link_libraries(LibtorchOpenCVTutorials PRIVATE

        -Wl,--no-as-needed
        -Wl,--whole-archive
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch_cuda.so"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch_cpu.so"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch.so"
        -Wl,--no-whole-archive

        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libc10_cuda.so"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libc10.so"

        # 1. CUDA Runtime (from your toolkit)
        "/usr/local/cuda/lib64/libcudart.so"

        # 2. CUDA Driver (The one usually missing - found in system libs)
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1"

        # 3. NVRTC (Compiler library)
        "/usr/local/cuda/lib64/libnvrtc.so"
        csv
        ${OpenCV_LIBS}

        pthread
        dl
        m
)


find_package(OpenCV CONFIG REQUIRED)
target_compile_options(LibtorchOpenCVTutorials PRIVATE -O3 -Wall)
set_target_properties(LibtorchOpenCVTutorials PROPERTIES
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/external/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
)



