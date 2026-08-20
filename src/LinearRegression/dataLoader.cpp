//
// Created by moinshaikh on 5/30/26.
//


#include"../../include/Util.h"
#include<torch/torch.h>
#include<doctest.hpp>
#include<csv.hpp>
#include<vector>
#include<string>
#include<string_view>
/*
struct datasets {
    torch::Tensor xTrain;
    torch::Tensor yTrain;
};

datasets convertIntoTensor(std::string_view path, datasets& data)
{
    csv::CSVFormat format;
    format.delimiter(',').no_header();
    csv::CSVReader reader(path, format);

    std::vector<csv::CSVRow> rows(reader.begin(), reader.end());

    // Read CSV data into vector of vectors of strings
    std::vector<std::vector<std::string>> csvData;
    for (auto & row : rows)
    {
        std::vector<std::string> currentRow;
        for (auto & cell : row) {
            currentRow.push_back(cell.get<std::string>());
        }
        csvData.push_back(currentRow);
    }

    // Number of rows and columns
    int numRows = csvData.size();
    int numCols = csvData.empty() ? 0 : csvData[0].size();

    // Separate features (all columns except last) and labels (last column)
    int numFeatures = numCols - 1;

    // Pre-allocate contiguous float buffers
    std::vector<float> featuresBuffer;
    featuresBuffer.reserve(numRows * numFeatures);
    std::vector<float> labelsBuffer;
    labelsBuffer.reserve(numRows);

    // Parse strings to floats and separate features from labels
    for (int row = 0; row < numRows; ++row)
    {
        for (int col = 0; col < numCols; ++col)
        {
            float val = 0.0f;
            if (!csvData[row][col].empty())
            {
                try {
                    val = std::stof(csvData[row][col]);
                } catch (...) {
                    val = 0.0f; // Handle conversion errors gracefully
                }
            }
            // Last column is the label/target, rest are features
            if (col == numCols - 1)
            {
                labelsBuffer.push_back(val);
            }
            else
            {
                featuresBuffer.push_back(val);
            }
        }
    }

    // Convert flat float buffers to torch tensors using from_blob + clone
    data.xTrain = torch::from_blob(featuresBuffer.data(), {numRows, numFeatures}, torch::kFloat32).clone();
    data.yTrain = torch::from_blob(labelsBuffer.data(), {numRows, 1}, torch::kFloat32).clone();
    return data;
}
*/

class MultiVariateLinearRegressionModelImpl : public torch::nn::Module {
private:
    torch::nn::Linear linear{nullptr};
public:
    MultiVariateLinearRegressionModelImpl() : linear(register_module("linear",torch::nn::Linear(3,1)))
    {}

    torch::Tensor forward(torch::Tensor x) {
        return linear->forward(x);
    }
};
TORCH_MODULE(MultiVariateLinearRegressionModel);


TEST_CASE("dataloaderFromcsvFile")
{
   datasets data;
    convertIntoTensor(std::string(DATASETS_DIR) + "/data-01-test-score.csv", data);
    torch::Tensor xTrain = data.xTrain;
    torch::Tensor yTrain = data.yTrain;

    // Print tensor details
    print("xTrain shape", xTrain.sizes());
    print("yTrain shape", yTrain.sizes());
    print("xTrain", xTrain);
    print("yTrain", yTrain);

    auto model  = MultiVariateLinearRegressionModel();
    auto optimizer = torch::optim::SGD(model->parameters(), 1e-5);
    int epochs =20;
    for (int  epoch = 0; epoch  < epochs; ++epoch)
    {

        auto prediction = model->forward(xTrain);
        auto cost = torch::nn::functional::mse_loss(prediction, yTrain);
        optimizer.zero_grad();
        cost.backward();
        optimizer.step();
        std::cout<<"epoch: "<< epoch <<" Cost : " <<cost.item<float>()<<"\n";
    }
}