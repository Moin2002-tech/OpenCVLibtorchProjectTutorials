//
// Created by moinshaikh on 6/20/26.
//


#include<torch/torch.h>
#include<Util.h>
#include<doctest.hpp>

TEST_CASE("SoftmaxClassification")
{
    torch::manual_seed(1);
    torch::Tensor t = torch::tensor({1,2,3},torch::kFloat);
    torch::Tensor Hypothesis = torch::nn::functional::softmax(t,0);
    print("Hypothesis",Hypothesis);
    print("SumofHypothesis",Hypothesis.sum());

    torch::Tensor z = torch::rand({3,5},torch::requires_grad(true));
    torch::Tensor Hypothesis2 = torch::nn::functional::softmax(z,1);
    print("Hypothesis2",Hypothesis2);

    torch::Tensor y = torch::randint(5,3,torch::kLong);

    auto yHotOne = torch::zeros_like(Hypothesis2);
    yHotOne.scatter_(1,y.unsqueeze(1),1);

    print("yHotOne",yHotOne);

    torch::Tensor cost =  (yHotOne*-torch::log(Hypothesis2)).sum(1).mean();
    print("Cost",cost);

    torch::Tensor cost2 = torch::nn::functional::cross_entropy(z,y);
    print("cost2",cost2);


}

TEST_CASE("LowLevelSoftmaxClassification") {
    torch::Tensor xTrain  = torch::tensor({{1, 2, 1, 1},
                                           {2, 1, 3, 2},
                                           {3, 1, 3, 4},
                                           {4, 1, 5, 5},
                                           {1, 7, 5, 5},
                                           {1, 2, 5, 6},
                                           {1, 6, 6, 6},
                                           {1, 7, 7, 7}},torch::kFloat);
    torch::Tensor yTrain = torch::tensor({2, 2, 2, 1, 1, 1, 0, 0},torch::kLong);

    torch::Tensor W = torch::zeros({4,3},torch::requires_grad(true));
    torch::Tensor b = torch::zeros({1},torch::requires_grad(true));

    auto optimizer = torch::optim::SGD({W,b},0.1);

    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; ++epoch) {

        auto hypothesis = torch::nn::functional::softmax(xTrain.matmul(W)+ b,1);
        auto yOneHot = torch::zeros_like(hypothesis);
        yOneHot.scatter_(1,yTrain.unsqueeze(1),1);
        auto cost = (yOneHot*-torch::log(torch::nn::functional::softmax(hypothesis,1))).sum(1).mean();

        optimizer.zero_grad();
        cost.backward();
        optimizer.step();


        if (epoch % 100 == 0)
        {
            std::cout<<"epoch: " <<epoch <<"\tcost:"<<cost.item<float>()<< " \n";
        }
    }
}

class softmaxClassificationImpl : public torch::nn::Module {
private:
    torch::nn::Linear linear{nullptr};
public:
    softmaxClassificationImpl() : linear(register_module("linear",torch::nn::Linear(4,3)))
    {}
    torch::Tensor forward(torch::Tensor x)
    {
        return linear->forward(x);
    }
};
TORCH_MODULE(softmaxClassification);

TEST_CASE("HighLevelSoftmaxClassification")
{
    torch::Tensor xTrain  = torch::tensor({{1, 2, 1, 1},
                                               {2, 1, 3, 2},
                                               {3, 1, 3, 4},
                                               {4, 1, 5, 5},
                                               {1, 7, 5, 5},
                                               {1, 2, 5, 6},
                                               {1, 6, 6, 6},
                                               {1, 7, 7, 7}},torch::kFloat);
    torch::Tensor yTrain = torch::tensor({2, 2, 2, 1, 1, 1, 0, 0},torch::kLong);

    auto model  = softmaxClassification();
    auto optimizer = torch::optim::SGD(model->parameters(),0.1);

    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto prediction = model->forward(xTrain);
        auto cost = torch::nn::functional::cross_entropy(prediction,yTrain);

        optimizer.zero_grad();
        cost.backward();
        optimizer.step();

        if (epoch % 100 == 0)
        {
            std::cout<<"epoch: " <<epoch <<"\tcost:"<<cost.item<float>()<< " \n";
        }
    }

}