//
// Created by moinshaikh on 6/5/26.
//


#include<doctest.hpp>
#include<torch/torch.h>
#include"../../include/Util.h"
#include<vector>
#include<string>
#include<string_view>

TEST_CASE("LogisticClassification")
{
    torch::manual_seed(1);
    torch::Tensor xTrain =  torch::tensor({{1,2},{2,3},{3,1},{4,3},{5,3},{6,2}},torch::kFloat);
    torch::Tensor yTrain =  torch::tensor({{0},{0},{0},{1},{1},{1}},torch::kFloat);
    //print("xTrain",xTrain);
    //print("yTrain",yTrain);
    print("xtrain shape",xTrain.sizes());
    print("yTrain shape",yTrain.sizes());
    //torch has torch::exp for resemble e exponesial
    print("e^1 equal to: ",torch::exp(torch::tensor(1.0)));

    auto w = torch::zeros({2,1},torch::requires_grad(true));
    auto b = torch::zeros({1},torch::requires_grad(true));
    auto hypothesis = 1 / (1 + torch::exp(-(xTrain.matmul(w) + b)));
    print("hypothesis",hypothesis);
    print("hypothesis shape",hypothesis.sizes());


    auto hypothesisSigmoid = torch::sigmoid(xTrain.matmul(w)+ b);
    print("hypothesisSigmoid",hypothesisSigmoid);
    print("hypothesisSigmoid shape",hypothesisSigmoid.sizes());

    //computing cost function low level
    auto losses = -(yTrain * torch::log(hypothesisSigmoid) + (1 -yTrain) * (1 - hypothesisSigmoid));
    print("losses: ", losses);
    auto cost = torch::mean(losses);
    print("cost : ", cost);

    auto cost2 = torch::nn::functional::binary_cross_entropy(hypothesisSigmoid,yTrain);
    print("cost2",cost2);


}

TEST_CASE("LogisticClassificationLowBinary") {
    torch::manual_seed(1);
    torch::Tensor xTrain = torch::tensor({{1, 2}, {2, 3}, {3, 1}, {4, 3}, {5, 3}, {6, 2}},torch::kFloat);
    torch::Tensor yTrain = torch::tensor({{0}, {0}, {0}, {1}, {1}, {1}},torch::kFloat);
    auto W = torch::zeros({2,1},torch::requires_grad(true));
    auto b = torch::zeros({1},torch::requires_grad(true));

    auto optimizer = torch::optim::SGD({W,b},1);
    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {

        auto hypothesis = torch::sigmoid(xTrain.matmul(W) + b);
        auto cost = -(yTrain * torch::log(hypothesis) +
             (1 - yTrain) * torch::log(1 - hypothesis)).mean();
        optimizer.zero_grad();
        cost.backward();
        optimizer.step();

        if (epoch % 100 == 0) {
            std::cout<<"epoch :"<< epoch << "\t cost: "<<cost.item<float>()<<"\n";
        }
    }
}

TEST_CASE("WithBinaryCrossEntropy")
{
    torch::manual_seed(1);
    torch::Tensor xTrain = torch::tensor({{1, 2}, {2, 3}, {3, 1}, {4, 3}, {5, 3}, {6, 2}},torch::kFloat);
    torch::Tensor yTrain = torch::tensor({{0}, {0}, {0}, {1}, {1}, {1}},torch::kFloat);

    auto W = torch::zeros({2,1},torch::requires_grad(true));
    auto b = torch::zeros({1},torch::requires_grad(true));

    auto Optimizer = torch::optim::SGD({W,b},1);

    int epochs = 100;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto Hypothesis = torch::sigmoid(xTrain.matmul(W)+ b);
        auto cost = torch::nn::functional::binary_cross_entropy(Hypothesis,yTrain);
        Optimizer.zero_grad();
        cost.backward();
        Optimizer.step();
        if (epoch % 10 == 0)
        {
            std::cout<<"epoch: "<<epoch<< "\t cost: "<<cost.item<float>()<<"\n";
        }
    }

}