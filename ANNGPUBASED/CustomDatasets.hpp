//
// Created by moinshaikh on 5/2/26.
//

#ifndef LIBTORCHOPENCVTUTORIALS_CUSTOMDATASETS_HPP
#define LIBTORCHOPENCVTUTORIALS_CUSTOMDATASETS_HPP
#include<torch/torch.h>


class CustomDatasets : public torch::data::Dataset<CustomDatasets>
{
private:
    torch::Tensor features;
    torch::Tensor labels;
public:
    CustomDatasets(torch::Tensor features, torch::Tensor labels);
    torch::Tensor getFeatures() const
    { return features; }
    torch::Tensor getLabels() const
    { return labels; }

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

};

#endif //LIBTORCHOPENCVTUTORIALS_CUSTOMDATASETS_HPP
