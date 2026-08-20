# LibtorchOpenCVTutorials

A comprehensive C++ project demonstrating machine learning and deep learning concepts using **LibTorch (PyTorch C++ API)**, **OpenCV**, and related libraries. Includes implementations of Linear Regression, Logistic Regression, Neural Networks (ANN, CNN), Autograd, Tensor operations, and more.

## Version

- **LibTorch**: 2.9.1 (CUDA 12.6)
- **OpenCV**: 4.x (installed via vcpkg)
- **C++ Standard**: C++20
- **CMake**: 3.20+
- **CUDA**: 12.6

---

## Prerequisites

Install the following dependencies **before** building the project.

### 1. Required System Packages

```bash
# Build tools
sudo apt update
sudo apt install build-essential cmake curl wget git pkg-config

# CUDA (must match LibTorch version - CUDA 12.6)
# Install NVIDIA CUDA Toolkit 12.6 from: https://developer.nvidia.com/cuda-downloads

# Required system libraries
sudo apt install libpthread-stubs0-dev libcurl4-openssl-dev
```

### 2. vcpkg (Package Manager)

```bash
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
```

### 3. Install vcpkg Packages

The project requires the following vcpkg packages (**LibTorch is NOT installed via vcpkg** - see next section):

```bash
cd ~/vcpkg

# Install required packages (listed individually for clarity)
./vcpkg install opencv4

./vcpkg install benchmark
./vcpkg install curl
./vcpkg install pthread
```

> **Note**: The `csv-parser` and `smartStore` libraries are included as git submodules in the `external/` directory and do not require separate installation.

### 4. Download & Link LibTorch (2.9.1 CUDA 12.6)

LibTorch is **not** installed via vcpkg. You must download and place it in the `external/libtorch/` directory manually.

```bash
# Navigate to the project root
cd /path/to/LibtorchOpenCVTutorials

# Create external directory if it doesn't exist
mkdir -p external

# Download LibTorch C++ distribution (CUDA 12.6 version)
wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.9.1%2Bcu126.zip

# Unzip directly into external/
unzip libtorch-cxx11-abi-shared-with-deps-2.9.1+cu126.zip -d external/

# Verify structure
ls external/libtorch/lib/libtorch.so   # should exist
```

**Directory structure after download:**
```
external/
  libtorch/
    include/         # LibTorch headers
    lib/             # LibTorch shared libraries (.so files)
    share/           # CMake config files
```

> **Important**: The CMakeLists.txt links directly to the `.so` files in `external/libtorch/lib/` using absolute paths. Ensure the directory structure matches the expected layout.

### 5. Configure vcpkg Toolchain in CMake

Ensure your CMake configuration uses the vcpkg toolchain file. Either set the `CMAKE_TOOLCHAIN_FILE` variable or use the `-D` flag during configuration.

**Option A: Environment variable**
```bash
export VCPKG_ROOT=~/vcpkg
```

**Option B: CMake configuration with toolchain**
```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
```

---

## Building the Project

```bash
# From the project root
mkdir -p build
cd build

# Configure with vcpkg toolchain
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake

# Build
make -j$(nproc)

# Run the executable
./LibtorchOpenCVTutorials
```

---

## Project Structure

```
├── CMakeLists.txt
├── main.cpp
├── include/
│   └── Util.h
├── databases/
│   ├── data-01-test-score.csv
│   ├── data-02-stock_daily.csv
│   ├── data-03-diabetes.csv
│   ├── data-04-zoo.csv
│   ├── fashion_minist_large.data
│   └── fmnist_small.csv
├── external/
│   ├── libtorch/                  # Manual download (NOT vcpkg)
│   ├── csv-parser/                # Git submodule
│   ├── smartStore/                # Git submodule
│   └── third_party/               # doctest
├── src/
│   ├── ANN/                       # ANN Fashion MNIST
│   ├── ANNGPUBASED/               # CNN & GPU-based implementations
│   ├── autogradsInLibtorch/       # Autograd examples
│   ├── BasicFunctionality/        # Tensor operations, Image reading
│   ├── LinearRegression/          # Single & Multi-variable Linear Regression
│   ├── LogisticRegression/        # Logistic Regression with data
│   ├── ReadingAndWriting/         # CSV reading, data validation
│   ├── TensorManipulation/        # Tensor manipulation utilities
│   └── TrainingPipeline/          # Neural Network training pipeline
```

---

## MNIST Dataset Download

The MNIST dataset files are **not included in the repository** due to their large size. If the binary files are not already present in `databases/mnist_data/`, download and extract them manually:

```bash
# From the project root
mkdir -p databases/mnist_data/raw && cd databases/mnist_data/raw

# Download the 4 files
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
if the following web not working
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz" -OutFile "train-images-idx3-ubyte.gz"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz" -OutFile "train-labels-idx1-ubyte.gz"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz" -OutFile "t10k-images-idx3-ubyte.gz"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz" -OutFile "t10k-labels-idx1-ubyte.gz"
# Extract
gunzip *.gz

# Move files up one directory (out of raw/)
mv *.ubyte ..
cd ..
rmdir raw
```

After extraction, your `databases/mnist_data/` folder should contain:
```
databases/mnist_data/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

---
TODO mnist back propagation
## Key Features

- **Linear Regression** (Single & Multi-variable)
- **Logistic Regression** (Binary classification)
- **Artificial Neural Networks** (ANN for Fashion MNIST)
- **Convolutional Neural Networks** (CNN with GPU support)
- **Autograd** in C++ (automatic differentiation)
- **Tensor operations & manipulation**
- **Data loading** from CSV files
- **Training pipeline** with custom data loaders

---

## License

This project is for educational purposes.
