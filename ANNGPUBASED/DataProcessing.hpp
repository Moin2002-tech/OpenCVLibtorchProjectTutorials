//
// Created by moinshaikh on 5/1/26.
//

#ifndef LIBTORCHOPENCVTUTORIALS_DATAPROCESSING_HPP
#define LIBTORCHOPENCVTUTORIALS_DATAPROCESSING_HPP

#include<csv.hpp>
#include<opencv2/opencv.hpp>
#include<torch/torch.h>

#include<string_view>

namespace DataProcessing
{

    struct Datasets
    {
        torch::Tensor X_train;
        torch::Tensor Y_train;
        torch::Tensor X_test;
        torch::Tensor Y_test;
    };
    struct TensorData
    {
      torch::Tensor X;
        torch::Tensor Y;
    };
    struct CSVData
    {
        std::vector<std::string> headers;
        std::vector<std::vector<std::string>> rows;
    };

    class DataProcessor
    {
    private:
        std::vector<csv::CSVRow> rawData;
        CSVData data;
        TensorData tensorData;
        Datasets datasets;
    public:
        explicit DataProcessor(std::string_view path,const csv::CSVFormat &format);
        void extractCSVData();
        void PreValidation();
        void visualData(
            int grid_size = 4,
            int original_img_size = 28,  // MNIST images are 28x28
            int display_img_size = 100,  // Scale up for better visibility
            int spacing = 15   // Space between images
            );
        CSVData getData() const { return data; }

        TensorData getTensorData() const { return tensorData; }

        Datasets getDatasets() const { return datasets; }

        void convertToTensor();

        void SplitData();







    };

}

#endif //LIBTORCHOPENCVTUTORIALS_DATAPROCESSING_HPP
