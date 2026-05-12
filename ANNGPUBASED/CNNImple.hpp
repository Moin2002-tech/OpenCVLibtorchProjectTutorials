//
// Created by moinshaikh on 5/2/26.
//

#ifndef LIBTORCHOPENCVTUTORIALS_CNNIMPLE_HPP
#define LIBTORCHOPENCVTUTORIALS_CNNIMPLE_HPP

#include<torch/torch.h>
class CNNImpl : public torch::nn::Module
{
private:
    torch::nn::Sequential features_ = {nullptr};
    torch::nn::Sequential classifier_ = {nullptr};
public:
    CNNImpl(int numFeatures);

    torch::Tensor forward(torch::Tensor x);

};
TORCH_MODULE(CNN);
#endif //LIBTORCHOPENCVTUTORIALS_CNNIMPLE_HPP
