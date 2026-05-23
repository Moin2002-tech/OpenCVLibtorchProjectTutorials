//
// Created by moinshaikh on 3/5/26.
//

#include<iostream>
#include<torch/torch.h>
#include"../../external/third_party/doctest.hpp"
using std::cout;


/*
 * 1. The Pointer Layer (at::Tensor)

The torch::Tensor you declare in your C++ code is actually a PIMPL (Pointer to Implementation). It is a very small object (typically 8–16 bytes) that lives on the Stack.

    It doesn't hold data.

    It doesn't hold the shape.

    It simply holds a reference-counted pointer to a TensorImpl.

2. The Metadata Layer (c10::TensorImpl)

This is where the "identity" of the tensor lives. When you call tensor.reshape() or tensor.to(torch::kCUDA), you are modifying or creating a new TensorImpl. It contains:

    Sizes & Strides: The logical geometry (e.g., {2,3} with strides {3,1}).

    Dtype: float32, int64, etc.

    Device: CPU or CUDA:0.

    Storage Pointer: A pointer to the actual raw data block.

3. The Data Layer (c10::StorageImpl)

This is the "Flat Blob." It has no shape. It is just a contiguous array of bytes.

    The Caching Allocator: To avoid the massive overhead of calling cudaMalloc (which is very slow), LibTorch uses a Caching Allocator. It maintains a pool of available memory blocks on your T600.

    Reference Counting: Multiple TensorImpl objects can point to the same StorageImpl. This is how "Views" work.

The "Dry Run" of a View Operation

Let’s look at what happens in memory when you do a "Zero-Copy" operation like a transpose or a slice.
C++

torch::Tensor a = torch::ones({2, 3}, device); // 1. Allocates Storage, creates Impl
torch::Tensor b = a.view({6, 1});              // 2. Creates NEW Impl, points to SAME Storage

What happened in VRAM?

    Storage: No new memory was allocated. The 6 ones stay exactly where they are.

    Tensor a: Has metadata Sizes: {2, 3}, Strides: {3, 1}.

    Tensor b: Has metadata Sizes: {6, 1}, Strides: {1, 1}.

When you access b[4][0], the math happens: (4×1)+(0×1)=4. It goes to the 4th index of the original storage. This is why view() is nearly instantaneous.
The "Storage Offset" Concept

There is one more hidden variable: Storage Offset. This allows a tensor to point to the middle of a memory block. This is how slicing works without copying.

If you have a large tensor and you take a slice:
C++

torch::Tensor big = torch::randn({10, 10}, device);
torch::Tensor small = big.slice(0, 5, 10); // Take rows 5 through 9

small now has a Storage Offset of 50 (assuming float32). Its pointer doesn't point to the start of the memory block; it points to the 50th element.
Summary Table: What lives where?
Component	Location	What it knows
torch::Tensor	Stack	"I am a handle."
TensorImpl	Heap	"I am 2x3, I am Float, I live on CUDA."
StorageImpl	GPU VRAM	"I am 24 bytes of raw data."
Why this matters for your C++ Development

Because you are working in C++, you have to be careful with lifetimes. If you create a "View" of a tensor and let the original tensor go out of scope, the Storage will stay alive as long as that "View" exists because of the reference counting in StorageImpl.
 *
 *
 */



TEST_CASE("BasicOperationOnTensors")
{
    torch::Device device = torch::kCUDA;
    //empty tensors.
    //generate garbage value only use when buffer or make it overwrite by data
    torch::Tensor tensor = torch::empty({2,3});
    //std::cout<< tensor;

    {
        torch::Tensor temp = torch::full({1000, 1000}, 99.9, device);
        // As soon as we hit this closing brace, 'temp' is destroyed.
        // BUT, PyTorch's caching allocator holds onto that VRAM block for future use.
    }

    // 2. Now allocate an "empty" tensor of the exact same size.
    // The caching allocator will recycle the exact same block of memory 'temp' just used.
    torch::Tensor recycled_empty = torch::empty({1000, 1000}, device);
    /*
    *
        Testing started at 11:25 PM ...
        99.9000  99.9000  99.9000
        99.9000  99.9000  99.9000
        [ CUDAFloatType{2,3} ]
     */

    // 3. Print a tiny 2x3 slice of it.
    std::cout << recycled_empty.slice(0, 0, 2).slice(1, 0, 3) << '\n';
    //using zeros
    torch::Tensor tensor2 = torch::zeros({2,3},torch::kCUDA);
 //   std::cout<<tensor2;

    auto tensorOnes = torch::ones({2,3},device);
    cout<<tensorOnes<<"\n";

    auto randomTensors = torch::rand({2,3},device);
    cout<<randomTensors<<"\n";

    //with the seed
    torch::manual_seed(100);
    auto tensorRandomSeed = torch::rand({2,3},device);

    torch::manual_seed(100);
    auto tensorRandomSeed2 = torch::rand({2,3},device);

    std::cout<<"seed 100 : "<<tensorRandomSeed<<"\n"<<"seed 100 : "<<tensorRandomSeed2<<"\n";

    //manual assigning a tensor using constructor
    torch::Tensor manualTensor = torch::tensor({{1,2,2},{3,6,4}},device);
    std::cout<<"manual Tensor Constructor"<<manualTensor<<"\n";

    //arange, full, eye, linespace

    auto arangedTensor = torch::arange(0,10,2);
    std::cout<<"arange Tensors: " <<arangedTensor<<"\n";
    auto fullTensor = torch::full({3,3},5);
    std::cout<<"full Tensors: "<<fullTensor<<"\n";
    auto linespaceTenosr = torch::linspace(0,10,10);
    std::cout<<"Linespace Tensor: "<<linespaceTenosr<<"\n";
    auto eyeTensor = torch::eye(5);
    std::cout<<"eye Tensor: "<<eyeTensor;

    std::cout<<"eyeTensorShape: "<< eyeTensor.sizes();

    std::cout<<"eyelikeemty"<<empty_like(eyeTensor);

    std::cout<<"eyelikeZeros"<<zeros_like(eyeTensor);

    std::cout<<"eyelikeOnes"<<ones_like(eyeTensor);

    std::cout<<"eyeLikeRandom"<<rand_like(eyeTensor);

    std::cout<<"eyeTensor Type : "<<eyeTensor.type()<<"\n"<<eyeTensor.dtype()<<"\n";

    auto tensorDatatypes = torch::tensor({{4,4,5,},{4,58,4}},torch::kInt32);
    std::cout<<tensorDatatypes.type()<<"\n";
   tensorDatatypes=  tensorDatatypes.to(torch::kFloat32);
    std::cout<<tensorDatatypes.type()<<"\n";
    /*
            * | **Data Type**             | **Dtype**         | **Description**                                                                                                                                                                |
            |---------------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            | **32-bit Floating Point** | `torch.float32`   | Standard floating-point type used for most deep learning tasks. Provides a balance between precision and memory usage.                                                         |
            | **64-bit Floating Point** | `torch.float64`   | Double-precision floating point. Useful for high-precision numerical tasks but uses more memory.                                                                               |
            | **16-bit Floating Point** | `torch.float16`   | Half-precision floating point. Commonly used in mixed-precision training to reduce memory and computational overhead on modern GPUs.                                            |
            | **BFloat16**              | `torch.bfloat16`  | Brain floating-point format with reduced precision compared to `float16`. Used in mixed-precision training, especially on TPUs.                                                |
            | **8-bit Floating Point**  | `torch.float8`    | Ultra-low-precision floating point. Used for experimental applications and extreme memory-constrained environments (less common).                                               |
            | **8-bit Integer**         | `torch.int8`      | 8-bit signed integer. Used for quantized models to save memory and computation in inference.                                                                                   |
            | **16-bit Integer**        | `torch.int16`     | 16-bit signed integer. Useful for special numerical tasks requiring intermediate precision.                                                                                    |
            | **32-bit Integer**        | `torch.int32`     | Standard signed integer type. Commonly used for indexing and general-purpose numerical tasks.                                                                                  |
            | **64-bit Integer**        | `torch.int64`     | Long integer type. Often used for large indexing arrays or for tasks involving large numbers.                                                                                  |
            | **8-bit Unsigned Integer**| `torch.uint8`     | 8-bit unsigned integer. Commonly used for image data (e.g., pixel values between 0 and 255).                                                                                    |
            | **Boolean**               | `torch.bool`      | Boolean type, stores `True` or `False` values. Often used for masks in logical operations.                                                                                      |
            | **Complex 64**            | `torch.complex64` | Complex number type with 32-bit real and 32-bit imaginary parts. Used for scientific and signal processing tasks.                                                               |
            | **Complex 128**           | `torch.complex128`| Complex number type with 64-bit real and 64-bit imaginary parts. Offers higher precision but uses more memory.                                                                 |
            | **Quantized Integer**     | `torch.qint8`     | Quantized signed 8-bit integer. Used in quantized models for efficient inference.                                                                                              |
            | **Quantized Unsigned Integer** | `torch.quint8` | Quantized unsigned 8-bit integer. Often used for quantized tensors in image-related tasks.                                                                                     |

     */

    //mathematical Operation on Tensors

    torch::Tensor tensorMath = torch::rand({4,3},device);
    cout<<"addition: "<<tensorMath + 2;
    cout<<" multiplication: "<<tensorMath *4;
    cout<<"divide: "<<tensorMath / 2;
    cout<<"minus"<<tensorMath - 2;

    torch::Tensor tensorMath2 = torch::rand({4,3},device);
    cout<<"addtion by elements of both tensors matrix: " <<tensorMath +tensorMath2<<"\n";
    cout<<"multplication by elements of the tensors matrix: "<<tensorMath * tensorMath2 <<"\n";
    cout<<"minus by elements of the tensors matrix: "<< tensorMath -tensorMath2<<"\n";
    cout<<"divide by elements of the tensors matrix: "<<tensorMath / tensorMath2 <<"\n";

    torch::Tensor tensorImpl = torch::tensor({4,-5,8,5},device);
    std::cout<<"orignal tensor: "<<tensorImpl;
    std::cout<<"abs function: "<< tensorImpl.abs();

    std::cout<<"negative become positve positive become negative: "<<tensorImpl.neg();

    torch::Tensor d = torch::tensor({1.0,5.0,4.0,1.1});

    std::cout<<"orginal d" <<d;
    std::cout<<"round"<<d.round();
    std::cout<<"ceil"<<d.ceil();
    std::cout<<"floor: "<<d.floor();
    std::cout<<"clamp:"<<d.clamp(2,3);


    //there is also an same functions with`_` underscore.
    /*
    * LibTorch provides a dual-interface for most tensor operations to balance safety and performance.
    * Standard methods (e.g., clamp()) implement Immutable Semantics,
    * returning a newly allocated tensor while preserving the source buffer.
    * This is the preferred approach for high-level logic and Autograd-tracked variables,
    * as it prevents side effects where modifying a derived "view" unintentionally alters the parent tensor.
    In-Place Optimization and Memory Re-use

    Methods suffixed with an underscore (e.g., clamp_()) implement Mutable Semantics.
    These functions operate directly on the existing c10::StorageImpl without triggering the Caching Allocator for a new buffer. W
    hile computationally efficient and recommended for low-level data processing or memory-bound loops,
    they should be used with caution when the tensor is part of a computational graph or shared across multiple handles.
     */

    /*
     * example
    torch::Tensor raw_sensor_data = torch::tensor({1.1, 5.8, 4.2});

        If you use the underscore here...
        auto processed = raw_sensor_data.round_();

        ...the original data is GONE.
        You can no longer calculate the 'error' or 'loss' between
         the raw data and the rounded data because both are now {1.0, 6.0, 4.0}.
     */
    int low = 0;
    int high =10;
    auto e = torch::randint(low,high,{2,3},dtype(torch::kFloat64));
    std::cout<<"\n"<<e;

    std::cout<<"\n"<<"sum of e: "<< torch::sum(e);

    std::cout<<"\n"<<"sum along with column:"<<torch::sum(e,0);
    cout<<"\n"<<"sum along wiht rows: "<<torch::sum(e, 1);


    std::cout<<"\n"<<"mean of e"<<torch::mean(e);
    std::cout<<"\n"<<"mean of e with column"<<torch::mean(e,0);
    std::cout<<"\n"<<"median of e"<<torch::median(e);


    std::cout<<"\n"<<"minimum of e"<<torch::min(e);
    std::cout<<"\n"<<"max of e"<<torch::max(e);


    //special of function of libtorch

    auto p = torch::randint(0,10,{2,3},dtype(torch::kFloat64));

    cout<<p;

    cout<<"log of p: " << torch::log(p);
    cout<<"exp of p: "<< torch::exp(p);
    cout<<"softmax of p: "<<torch::softmax(p,0);
    cout<<"sigmoid of p" <<torch::sigmoid(p);

    cout<<"square root of p: "<<torch::sqrt(p);


    cout<<"the relu of p: "<< torch::relu(p);

    auto m = torch::rand({2,3});
    auto n = torch::rand({2,3});

    cout<<m<<"\n"<<n<<"\n";



}