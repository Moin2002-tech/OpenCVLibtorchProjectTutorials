//
// Created by moinshaikh on 6/19/26.
//


#include<csv.hpp>
#include<Util.h>
#include<torch/torch.h>
#include<doctest.hpp>

class BinaryClassificationImpl : public torch::nn::Module {
private:
    torch::nn::Linear linear{nullptr};
    torch::nn::Sigmoid sigmoid{nullptr};
public:
    BinaryClassificationImpl() : linear(register_module("linear",torch::nn::Linear(8,1))), sigmoid(register_module("sigmoid", torch::nn::Sigmoid()))
    {

    }
    torch::Tensor forward(torch::Tensor x) {
        return sigmoid->forward(linear->forward(x));
    }
};
TORCH_MODULE(BinaryClassification);


TEST_CASE("LogisticClassificationWithRealData")
{
    std::string_view path = "/home/moinshaikh/CLionProjects/LibtorchOpenCVTutorials/databases/data-03-diabetes.csv"  ;
    datasets data;
    convertIntoTensor(path, data);
    torch::Tensor xTrain = data.xTrain.to(torch::kFloat32);
    torch::Tensor yTrain = data.yTrain.to(torch::kFloat32);

    auto Model = BinaryClassification();
    auto Optimizer = torch::optim::SGD(Model->parameters(),1);
    int epochs = 100;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto Hypothesis = Model->forward(xTrain);
        auto Cost = torch::nn::functional::binary_cross_entropy(Hypothesis,yTrain);
        Optimizer.zero_grad();
        Cost.backward();
        Optimizer.step();

        if (epoch % 10 == 0)
        {
            torch::Tensor prediction = torch::ge(Hypothesis, 0.5);
            torch::Tensor correctPrediction = torch::eq(prediction.to(torch::kFloat), yTrain);
            float accuracy = correctPrediction.sum().item<float>() / correctPrediction.numel();
            printf("Epoch %4d/%d Cost: %.6f Accuracy %5.2f%%\n",
                   epoch, epochs, Cost.item<float>(), accuracy * 100);
        }

    }
}
