//
// Created by moinshaikh on 5/24/26.
//

#include<doctest.hpp>
#include<iostream>
#include<torch/torch.h>
#include<torch/nn.h>
#include"../../include/Util.h"
TEST_CASE("MinimaziingCost")
{
    print("______________MinimazingCost__________");

    torch::manual_seed(1);
    torch::Tensor X_train = torch::tensor({{1},{2},{3}},torch::kFloat);
    torch::Tensor Y_train = torch::tensor({{1},{2},{3}},torch::kFloat);
    torch::Tensor W = torch::zeros(1,torch::kFloat);
    //gradient decent
    torch::Tensor gradient = torch::sum((W * X_train -Y_train) * X_train);
    print("gradient", gradient);
    float lr = 0.01;
     W -= lr * gradient;
    print("W" , W);

}

TEST_CASE("Training")
{
    torch::manual_seed(1);
        print("_____training______");
    torch::Tensor X_train = torch::tensor({{1},{2},{3}},torch::kFloat);
    torch::Tensor Y_train = torch::tensor({{1},{2},{3}},torch::kFloat);
    torch::Tensor W = torch::zeros(1,torch::kFloat);
    float lr = 0.01;
    int epochs = 10;
    for (int epoch = 0; epoch< epochs; ++epoch) {
        auto hypothesis = W * X_train;

        //cost gradient
        auto cost =  torch::mean(torch::square(hypothesis - Y_train));
        auto gradient = torch::sum((W * X_train -Y_train) * X_train);
        std::cout<<"epoch: "<< epoch <<" "<< "W : "<< W.item()<<" Cost : " <<cost.item<float>()<<"\n";
        W -= lr * gradient;
    }
}

TEST_CASE("TrainingWithOptim")
{
    torch::Tensor X_train = torch::tensor({{1},{2},{3}},torch::kFloat);
    torch::Tensor Y_train = torch::tensor({{1},{2},{3}},torch::kFloat);
    float lr = 0.15;
    auto W = torch::zeros(1,torch::requires_grad(true));
    auto optimizer = torch::optim::SGD({W},lr);

    int epochs = 10;
    for (int epoch = 0; epoch < epochs ; ++epoch)
    {
        auto Hypothesis = X_train  *W;
        auto result = Hypothesis - Y_train;
        auto cost = torch::mean(torch::square(result));


        std::cout<<"epoch: "<< epoch <<" "<< "W : "<< W.item()<<" Cost : " <<cost.item<float>()<<"\n";
        optimizer.zero_grad();
        cost.backward();
        optimizer.step();
    }

}