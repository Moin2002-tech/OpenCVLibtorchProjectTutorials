//
// Created by moinshaikh on 7/22/26.
//


#include<torch/torch.h>
#include<Util.h>


#include<doctest.hpp>

TEST_CASE("xorNN")
{
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::manual_seed(777);
    if (device == torch::kCUDA) {
        torch::cuda::manual_seed_all(777);
    }
    torch::Tensor X = torch::tensor({{0,0},{0,1},{1,0}, {1,1}}).to(device).to(torch::kFloat);
    torch::Tensor Y = torch::tensor({{0},{1},{1},{0}}).to(device).to(torch::kFloat);
    auto linear = torch::nn::Linear(2,2);
    auto linear2 = torch::nn::Linear(2,1);
    auto sigmoid = torch::nn::Sigmoid();
    auto model = torch::nn::Sequential(linear,sigmoid,linear2,sigmoid);
    model->to(device);

    auto criterion = torch::nn::BCELoss();
    criterion->to(device);
    auto optimizer = torch::optim::SGD(model->parameters(),1);

    int epochs = 10001;
    for (int epoch = 0; epoch < epochs ;++epoch)
    {
        optimizer.zero_grad();
        auto hypothesis = model->forward(X);
        auto cost = criterion(hypothesis,Y);
        cost.backward();
        optimizer.step();

        if (epoch % 100 == 0) {
            std::cout << "Epoch: " << epoch << ", Cost: " << cost.item<double>() << std::endl;
        }

    }

    {
        torch::NoGradGuard no_grad;
        auto hypothesis = model->forward(X);
        auto predicted = torch::gt(hypothesis, 0.5).to(torch::kFloat);
        auto accuracy = torch::eq(predicted, Y).to(torch::kFloat).mean();
        std::cout << "\nHypothesis: " << hypothesis << "\nCorrect: " << predicted << "\nAccuracy: " << accuracy.item<double>() << std::endl;
    }


}