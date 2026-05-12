//
// Created by moinshaikh on 5/2/26.
//


#include "CNNImple.hpp"
#include<torch/nn.h>
CNNImpl::CNNImpl(int input_features) : torch::nn::Module("CNN")
{
    features_ = torch::nn::Sequential();
    features_->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(input_features, 32, 3).padding(1)));
    features_->push_back(torch::nn::ReLU());
    features_->push_back(torch::nn::BatchNorm2d(32));
    features_->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    
    features_->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
    features_->push_back(torch::nn::ReLU());
    features_->push_back(torch::nn::BatchNorm2d(64));
    features_->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    
    classifier_ = torch::nn::Sequential();
    classifier_->push_back(torch::nn::Flatten());
    classifier_->push_back(torch::nn::Linear(64 * 7 * 7, 128));
    classifier_->push_back(torch::nn::ReLU());
    classifier_->push_back(torch::nn::Dropout(0.4));
    
    classifier_->push_back(torch::nn::Linear(128, 64));
    classifier_->push_back(torch::nn::ReLU());
    classifier_->push_back(torch::nn::Dropout(0.4));
    
    classifier_->push_back(torch::nn::Linear(64, 10));
    
    register_module("features", features_);
    register_module("classifier", classifier_);
}

torch::Tensor CNNImpl::forward(torch::Tensor x) {
    x = features_->forward(x);
    x = classifier_->forward(x);
    return x;
}
