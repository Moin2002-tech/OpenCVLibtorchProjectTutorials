
//
// Created by moinshaikh on 5/23/26.
//
#include<iostream>
#include<torch/torch.h>
#include<doctest.hpp>
#include<torch/nn.h>
#include<torch/optim.h>


void println(const std::string& label) {
    std::cout << label << ":\n\n";
}

template <typename T>
void println(const std::string& label, const T& t)
{
    std::cout << label << ":\n" << t << "\n\n";
}

// Overload for torch::Tensor::max(dim) which returns (values, indices) tuple
void println(const std::string& label, const std::tuple<torch::Tensor, torch::Tensor>& t) {
    std::cout << label << ":\n";
    std::cout << "  values:  " << std::get<0>(t) << "\n";
    std::cout << "  indices: " << std::get<1>(t) << "\n\n";
}

TEST_CASE("LinearRegression")
{
    println("______LinearRegression_______");

    torch::manual_seed(1);
    auto X_train = torch::tensor({{1},{2},{3}},torch::kFloat);
    auto Y_train = torch::tensor({{1},{2},{3}},torch::kFloat);

    println("X_train",X_train);
    println("shape of X_train",X_train.sizes());


    auto W = torch::zeros(1,torch::requires_grad(true));
    println("weight" , W);

    auto b = torch::zeros(1,torch::requires_grad(true));
    println("bias",b);

        //hypothesis
    auto Hypothesis = X_train * W + b;
    println("hypothesis",Hypothesis);

    // cost

    println("cost", torch::square((Hypothesis - Y_train)));
   auto cost =  torch::mean(torch::square(Hypothesis - Y_train));

    //gradient Decent
    float learninRate = 0.01;
    auto optimizer =  torch::optim::SGD({W, b},learninRate );

    optimizer.zero_grad();
    cost.backward();
    optimizer.step();

    println("W", W);
    println("b", b);

}

TEST_CASE("TrainingLoopOfLinearRegression")
{
    auto X_train = torch::tensor({{1},{2},{3}},torch::kFloat);
    auto Y_train = torch::tensor({{1},{2},{3}},torch::kFloat);

    auto W = torch::zeros(1,torch::requires_grad(true));
    auto b = torch::zeros(1,torch::requires_grad(true));
    float learning = 0.01;
    auto optimezer = torch::optim::SGD({W, b}, learning);

    int epochs = 1000;
    for (int epoch = 0 ; epoch < epochs; ++epoch)
    {
        auto Hypothesis =X_train * W + b;
        auto cost = torch::mean(torch::square(Hypothesis - Y_train));
        optimezer.zero_grad();
        cost.backward();
        optimezer.step();

        if (epoch % 100 == 0)
        {
            std::cout<<"epoch: "<< epoch <<" "<< "W : "<< W.item()<<" b : "<< b.item()<< " Cost : " <<cost.item<float>()<<"\n";
        }
    }
}
// Equivalent of Python's:
//   class LinearRegressionModel(nn.Module):
//       def __init__(self):
//           super().__init__()
//           self.linear = nn.Linear(1, 1)
//       def forward(self, x):
//           return self.linear(x)
class LinearRegressionModelImpl : public torch::nn::Module
{
public:
    LinearRegressionModelImpl()
        : linear(register_module("linear", torch::nn::Linear(1, 1)))
    {}

    torch::Tensor forward(torch::Tensor x) {
        return linear->forward(x);
    }

private:
    torch::nn::Linear linear{nullptr};
};
TORCH_MODULE(LinearRegressionModel);


TEST_CASE("HighLevelLRmodel") {
    println("______HighLevelLinearRegression_______");

    // Create model — TORCH_MODULE wrapper, use value semantics
    LinearRegressionModel model;

    // Data
    auto X_train = torch::tensor({{1},{2},{3}}, torch::kFloat);
    auto Y_train = torch::tensor({{1},{2},{3}}, torch::kFloat);

    // Optimizer — TORCH_MODULE wrapper overloads -> to access the module
    torch::optim::SGD optimizer(model->parameters(), 0.01);

    // Training loop
    for (int epoch = 0; epoch < 1000; ++epoch) {
        auto prediction = model->forward(X_train);
        auto loss = torch::mean(torch::square(prediction - Y_train));

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch
                      << " | Loss: " << loss.item<float>() << "\n";
        }
    }

    // Predict
    auto test = torch::tensor({{4.0f}}, torch::kFloat);
    auto pred = model->forward(test);
    println("Prediction for x=4", pred);
}
