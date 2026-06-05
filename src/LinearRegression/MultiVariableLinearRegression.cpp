//
// Created by moinshaikh on 5/25/26.
//
#include<torch/torch.h>
#include<torch/nn.h>
#include<doctest.hpp>


#include"../../include/Util.h"

TEST_CASE("MultiVariableLinearRegression")
{
    torch::manual_seed(2);
    torch::Tensor X1_train = torch::tensor({{73}, {93},{89}, {96}, {73}},torch::kFloat);
    torch::Tensor X2_train = torch::tensor({{83}, {73},{69}, {98}, {83}},torch::kFloat);
    torch::Tensor X3_train = torch::tensor({{83}, {73},{70}, {98}, {83}},torch::kFloat);

    torch::Tensor Y_train = torch::tensor({{152},{105},{133},{100},{183}}, torch::kFloat);

    torch::Tensor w1 = torch::zeros(1,torch::requires_grad(true));
    torch::Tensor w2 = torch::zeros(1,torch::requires_grad(true));
    torch::Tensor w3 = torch::zeros(1,torch::requires_grad(true));
    auto b = torch::zeros(1,torch::requires_grad(true));
    float lr = 1e-5f;
    auto optimizer = torch::optim::SGD({w1,w2,w3,b}, lr);
    unsigned int epochs = 1000;
    for (unsigned int epoch = 0; epoch < epochs; ++epoch)
    {
        auto hypothesis = X1_train * w1 + X2_train * w2 + X3_train * w3 + b;
        auto cost = torch::mean(torch::square(hypothesis - Y_train));
        optimizer.zero_grad();
        cost.backward();
        optimizer.step();
        if (epoch % 1000 == 0)
        {
            std::cout << "epoch: " << epoch
                      << " W1: " << w1.item<float>()
                      << " W2: " << w2.item<float>()
                      << " W3: " << w3.item<float>()
                      << " b: " << b.item<float>()
                      << " Cost: " << cost.item<float>() << "\n";
        }
    }
}

class MultiVariableLinearRegressionImpl : public torch::nn::Module
{
public:
    MultiVariableLinearRegressionImpl() : linear1(register_module("linear1", torch::nn::Linear(3,1)))
    {

    }
    torch::Tensor forward(torch::Tensor X)
    {
       return linear1->forward(X);
    }
private:
    torch::nn::Linear linear1 {nullptr};
};
TORCH_MODULE(MultiVariableLinearRegression);

TEST_CASE("MultiVariableLinearRegressionHighLevel")
{

    print("__________________MultiVariableLinearRegressionHighLevel______________________-");
    torch::Tensor X_train = torch::tensor({{73, 80, 75},
                             {93, 88, 93},
                             {89, 91, 90},
                             {96, 98, 100},
                             {73, 66, 70}}, torch::kFloat);
    torch::Tensor Y_train = torch::tensor({{152}, {185}, {180}, {196}, {142}},torch::kFloat);

    float lr = 1e-5f;
    MultiVariableLinearRegression model;
    auto optimizer = torch::optim::SGD(model->parameters(),lr);
     int epochs = 40;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto predication = model->forward(X_train);
        auto cost =torch::nn::functional::mse_loss(predication, Y_train);
        optimizer.zero_grad();
        cost.backward();
        optimizer.step();

        std::cout<<"epoch: "<< epoch <<" Cost : " <<cost.item<float>()<<"\n";
    }
}