cmake_minimum_required(VERSION 3.20)
project(LibtorchOpenCVTutorials)



set(CMAKE_CXX_STANDARD 17)

# Configure doctest
if(MSVC)
    add_compile_options(/utf-8)
endif()
set(CMAKE_TOOLCHAIN_FILE "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain")
set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv4")


add_definitions(-DDATASETS_DIR="${CMAKE_SOURCE_DIR}/databases")
add_subdirectory(external/csv-parser)
find_package(OpenCV CONFIG REQUIRED)

add_executable(LibtorchOpenCVTutorials main.cpp
        external/third_party/doctest.cpp
        external/third_party/doctest.hpp
        src/BasicFunctionality/readingImage.cpp
        src/BasicFunctionality/BasicOperationOnTensors.cpp
        src/autogradsInLibtorch/autogradsincpp.cpp
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
        src/LinearRegression/minimazingCost.cpp
        include/Util.h
        src/LinearRegression/MultiVariableLinearRegression.cpp
        src/LinearRegression/dataLoader.cpp
        src/LogisticRegression/LogisticRegression.cpp
        src/LogisticRegression/LClassificationWithData.cpp
        src/softmaxClassification/SoftmaxClassification.cpp
        src/MNISTIntro/dataloading.cpp
)

target_link_libraries(LibtorchOpenCVTutorials PRIVATE)

target_include_directories(LibtorchOpenCVTutorials PRIVATE
        "${CMAKE_SOURCE_DIR}/external/libtorch/include"
        "${CMAKE_SOURCE_DIR}/external/libtorch/include/torch/csrc/api/include"
        "${OpenCV_INCLUDE_DIRS}"
        ${PROJECT_SOURCE_DIR}/external/third_party
        ${PROJECT_SOURCE_DIR}/include

)

target_link_libraries(LibtorchOpenCVTutorials PRIVATE

        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch.lib"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch_cpu.lib"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/c10.lib"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/fmt.lib"


        csv
        ${OpenCV_LIBS}
)


find_package(OpenCV CONFIG REQUIRED)
target_compile_options(LibtorchOpenCVTutorials PRIVATE -O3 -Wall)
set_target_properties(LibtorchOpenCVTutorials PROPERTIES
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/external/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
)



# Copy required DLLs next to the executable (RPATH does not work on Windows)
if(WIN32)
    add_custom_command(TARGET LibtorchOpenCVTutorials POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch_cpu.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/c10.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch_global_deps.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/uv.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/shm.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libiomp5md.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libiompstubs5md.dll"
            "$<TARGET_FILE_DIR:LibtorchOpenCVTutorials>"
    )
endif()