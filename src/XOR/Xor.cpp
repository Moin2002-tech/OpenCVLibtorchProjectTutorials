//
// Created by moinshaikh on 7/10/26.
//


#include<torch/torch.h>
#include<doctest.hpp>

TEST_CASE("XOR")
{
    torch::Device device =  torch::cuda::cudnn_is_available() ? torch::kCUDA :  torch::kCPU;
    torch::manual_seed(777);
    if (device==torch::kCUDA) {
        torch::cuda::manual_seed_all(777);
    }
    torch::Tensor X = torch::tensor({{0,0},{0,1},{1,0},{1,1}},torch::TensorOptions(device).dtype(torch::kFloat));
    torch::Tensor Y = torch::tensor({{0},{1},{1},{0}},torch::TensorOptions(device).dtype(torch::kFloat));

    auto linear = torch::nn::Linear(2,1);
    auto sigmoid = torch::nn::Sigmoid();

    auto model = torch::nn::Sequential(linear,sigmoid);

    model->to(device);

    auto critic = torch::nn::BCELoss();
    critic->to(device);
    auto optimizer = torch::optim::SGD(model->parameters(),1);

    uint epochs = 1001;

    for (uint epoch = 0; epoch < epochs; epoch++) {
        optimizer.zero_grad();
        auto hypthesis = model->forward(X);
        auto cost = critic->forward(hypthesis, Y);
        cost.backward();
        optimizer.step();
        if (epoch % 100 == 0) {
            std::cout<<"Epoch "<<epoch<< " \t" <<cost.item<double>()<<std::endl;
        }
    }

    // Evaluation with no_grad
    {
        torch::NoGradGuard no_grad;
        auto hypothesis = model->forward(X);
        auto predicted = torch::gt(hypothesis, 0.5).to(torch::kFloat);
        auto accuracy = torch::eq(predicted, Y).to(torch::kFloat).mean();
        std::cout << "\nHypothesis: " << hypothesis << "\nCorrect: " << predicted << "\nAccuracy: " << accuracy.item<double>() << std::endl;
    }
}