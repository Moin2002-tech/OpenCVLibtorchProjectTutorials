//
// Created by moinshaikh on 5/2/26.
//

#include"CustomDatasets.hpp"
CustomDatasets::CustomDatasets(torch::Tensor features, torch::Tensor labels)
{
    // Convert to PyTorch tensors with proper data types
    this->features = features.to(torch::kFloat32).reshape({-1, 1, 28, 28});
    this->labels = labels.to(torch::kLong);
}

torch::data::Example<> CustomDatasets::get(size_t index)
{
    return {features[index], labels[index]};
}

torch::optional<size_t> CustomDatasets::size() const
{
    return features.size(0);
}
