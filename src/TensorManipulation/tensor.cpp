//
// Created by moinshaikh on 5/22/26.
//
#include "doctest.hpp"

#include <torch/torch.h>
#include <iostream>

template <typename T>
void print(const std::string& label, const T& t) {
    std::cout << label << ":\n" << t << "\n\n";
}

// Overload for torch::Tensor::max(dim) which returns (values, indices) tuple
void print(const std::string& label, const std::tuple<torch::Tensor, torch::Tensor>& t) {
    std::cout << label << ":\n";
    std::cout << "  values:  " << std::get<0>(t) << "\n";
    std::cout << "  indices: " << std::get<1>(t) << "\n\n";
}

TEST_CASE("tensorManipulation")
{
    torch::Tensor tensor = torch::tensor({4,4,55,4,5,5,45,545});
    print("1D tensor", tensor);
}

TEST_CASE("tensorSlicing")
{
    // --- 1D Tensor Slicing ---
    torch::Tensor t1d = torch::arange(10);
    print("t1d", t1d);

    // index(): single element
    print("t1d.index({0})", t1d.index({0}));          // 0
    print("t1d.index({-1})", t1d.index({-1}));         // 9 (last)

    // slice(start, stop, step)
    auto slice = [](const torch::Tensor& t, int64_t start, int64_t stop, int64_t step) {
        return t.index({torch::indexing::Slice(start, stop, step)});
    };

    print("t1d[2:7]",     t1d.index({torch::indexing::Slice(2, 7)}));       // 2..6
    print("t1d[:5]",      t1d.index({torch::indexing::Slice(NULL, 5)}));    // 0..4
    print("t1d[5:]",      t1d.index({torch::indexing::Slice(5, NULL)}));    // 5..9
    print("t1d[::2]",     t1d.index({torch::indexing::Slice(NULL, NULL, 2)})); // even indices

    // --- 2D Tensor Slicing ---
    torch::Tensor t2d = torch::arange(20).reshape({4, 5});
    print("t2d (4x5)", t2d);

    // Single row: t2d[1]
    print("t2d[1]", t2d.index({1}));

    // Single col: t2d[:, 2]
    print("t2d[:, 2]", t2d.index({torch::indexing::Slice(), 2}));

    // Block: t2d[1:3, 2:4]
    print("t2d[1:3, 2:4]", t2d.index({
        torch::indexing::Slice(1, 3),
        torch::indexing::Slice(2, 4)
    }));

    // First 2 rows, all cols
    print("t2d[:2]", t2d.index({torch::indexing::Slice(NULL, 2)}));

    // --- Indexing with tensors (advanced) ---
    torch::Tensor indices = torch::tensor({0, 2, 3});
    print("t2d.index({0,2,3}) rows", t2d.index({indices}));

    // --- Boolean masking ---
    torch::Tensor mask = t2d > 10;
    print("mask (t2d > 10)", mask);
    print("t2d[mask]", t2d.index({mask}));

    // --- Narrow (equivalent to slice but uses size instead of stop) ---
    // narrow(dim, start, length)
    print("t2d.narrow(0, 1, 2)", t2d.narrow(0, 1, 2));   // rows 1..2
    print("t2d.narrow(1, 2, 3)", t2d.narrow(1, 2, 3));   // cols 2..4

    // --- Select (picks one index along a dim, dropping that dim) ---
    print("t2d.select(0, 2)", t2d.select(0, 2));  // row 2
    print("t2d.select(1, 3)", t2d.select(1, 3));  // col 3
}


TEST_CASE("ArthimeticOperationOnTensors") {
    torch::Tensor tensor1d1 = torch::arange(20).reshape({4,5});
    torch::Tensor tensor2d1 = torch::arange(20).reshape({4,5});
    print("tensor1d1", tensor1d1);
    print("tensor2d1",tensor2d1);
    print("tensor1d X tensor2d",tensor1d1 *tensor2d1);

    torch::Tensor tensorA = torch::tensor({{5,4},{4,6}});
    torch::Tensor tensorB = torch::tensor({{4},{3}});
    print("tensorA",tensorA);
    print("tensorB",tensorB);
    print("shape Of TensorA",tensorA.sizes());
    print("shape Of TensorB",tensorB.sizes());

    print("matmul A to B",tensorA.matmul(tensorB));
    print("* multiplication",tensorA * tensorB);

    print("mul",tensorA.mul(tensorB));
    torch::Tensor floatTensor =  torch::tensor({{2,2},{4,4}},torch::kFloat);
    print("mean", floatTensor.mean());
    /// you can't mean of the integers tensors
    torch::Tensor intTensor = torch::tensor({{2,2},{4,4}},torch::kInt);


    /*
    try
    {
        print("taking int Tensor mean", intTensor.mean());
    }
    catch (const std::exception &e) {
        std::cout<<e.what()<<"\n";
    }*/
    print("with dim 0",NULL);
    print("mean with dim 0 ",floatTensor.mean(0));
    print("mean with dim 1", floatTensor.mean(1));
    print("mean with dim -1",floatTensor.mean(-1));

    print("Sum 2d matrix",NULL);
    print("Sum 2d matrix" ,floatTensor.sum());
    print("Sum 2d matrix dim 1" ,floatTensor.sum(1));
    print("Sum 2d matrix dim -1" ,floatTensor.sum(-1));
    print("Sum 2d matrix dim 0" ,floatTensor.sum(0));


    print("ArgMAx and max in tensors", NULL);
    torch::Tensor argsTensor = torch::tensor({{1,2},{3,4}}, torch::kFloat);
    print("argsTensor",argsTensor);
    print("max tensor",  argsTensor.max());
    print("max tensor argmax",argsTensor.max(0));
    print("argmaz 1",argsTensor.argmax(1));


    print("torch::view",NULL);

    torch::Tensor viewTensor =  torch::tensor({{0,1,2},
                                                {3,4,5,},
                                                {6,7,8},
                                                {9,10,11}}, torch::kFloat);
    print("viewTensor",viewTensor);

    print("dim",viewTensor.sizes());


    print("[-1,3]",viewTensor.view({-1,3}));
    print("[2,1,-1]",viewTensor.view({2,1,-1}));
    print("[3,1,-1]",viewTensor.view({3,1,-1}));

    torch::Tensor ft = torch::tensor({{0},{1},{2}},torch::kFloat);
    print("ft",ft);
    print("shape of ft",ft.sizes());
    print("squeeze" , ft.squeeze());
    print("shape of squeezed",ft.squeeze().sizes());




}