//
// Created by moinshaikh on 3/13/26.
//


#include<map>
#include<iostream>
#include<string>
#include<vector>
#include<torch/torch.h>
#include"../../external/third_party/doctest.hpp"
#include"csv.hpp"




class NNModelImpl : public torch::nn::Module
{
public:
    torch::nn::Linear linear{nullptr};
    torch::nn::Sigmoid sigmoid{nullptr};

    NNModelImpl(int features) : linear(torch::nn::Linear(features, 1)), sigmoid(torch::nn::Sigmoid()) {
        register_module("linear", linear);
        register_module("sigmoid", sigmoid);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto out = linear(x);
        out = sigmoid(out);
        return out;
    }
};
TORCH_MODULE(NNModel);
TEST_CASE("TrainingPipelineNNModule")
{
    std::string path = "/home/moinshaikh/CLionProjects/LibtorchOpenCVTutorials/databases/data.csv";
    csv::CSVFormat format;
    format.delimiter(',').no_header();

    csv::CSVReader reader(path, format);
    std::vector<csv::CSVRow> rows(reader.begin(),reader.end());

    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> csvData;
    bool firstRow = true;
    for (auto & row : rows)
    {
        std::vector<std::string> currentRow;
        for (auto & cell : row)
        {
            currentRow.push_back(cell.get<std::string>());
        }
        if (firstRow)
        {
            headers = currentRow;
            firstRow = false;
        }
        else
        {
            csvData.push_back(currentRow);
        }
    }

    //Pre Validation
    size_t targetColumnCount = headers.size();
    for (auto & row : csvData)
    {
        if (row.size() < targetColumnCount)
        {
            row.resize(targetColumnCount,"0.0");
        }
        else if (row.size() > targetColumnCount)
        {
            row.erase(row.begin() + targetColumnCount, row.end());
        }
    }



    std::map<std::string,torch::Tensor> headersTensors;
    int rowCount =  csvData.size();

    for (int col =0 ; col < headers.size(); ++col)
    {
        std::string name = headers[col];
        if (name == "id")
            continue;

        headersTensors[name] = torch::zeros({rowCount},torch::kFloat32);
        for (int row = 0; row < rowCount; ++row)
        {
            std::string cell_value = csvData[row][col];
            if (name == "diagnosis")
            {
                // Label encoding: M->1, B->0
                headersTensors[name][row] = (cell_value == "M") ? 1.0f : 0.0f;
            }
            else
            {
                try
                {
                    headersTensors[name][row] = std::stof(cell_value);
                }
                catch (...)
                {
                    headersTensors[name][row] = 0.0f;
                }
            }
        }
    }

    // --- STEP 4: CREATE FEATURE MATRIX AND TARGET VECTOR ---
    // Stack all feature tensors (exclude diagnosis)

    std::vector<torch::Tensor> featureTensors;
    for (auto & [name, tensor] : headersTensors)
    {
        if (name != "diagnosis")
        {
            featureTensors.push_back(tensor);
        }
    }
    // Create feature matrix X and target y
    auto X = torch::stack(featureTensors,1);
    auto Y = headersTensors["diagnosis"];




    //NORMALISE
    // Normalize features (zero mean, unit variance)
    auto X_mean = X.mean(0, true);
    auto X_std  = X.std(0, true);
    X_std = torch::clamp(X_std, 1e-7);
    X = (X - X_mean) / X_std;

    std::cout << "Feature matrix shape: " << X.sizes() << "\n";
    std::cout << "Target vector shape: "  << Y.sizes() << "\n";

    // --- STEP 5: TRAIN/TEST SPLIT ---
    int total_samples = X.size(0);
    int test_size     = static_cast<int>(total_samples * 0.2);
    int train_size    = total_samples - test_size;

    auto indices       = torch::randperm(total_samples);
    auto train_indices = indices.slice(0, 0, train_size);
    auto test_indices  = indices.slice(0, train_size, total_samples);

    auto X_train = X.index_select(0, train_indices);
    auto X_test  = X.index_select(0, test_indices);
    auto y_train = Y.index_select(0, train_indices).unsqueeze(1);
    auto y_test  = Y.index_select(0, test_indices).unsqueeze(1);

    std::cout << "Training set size: " << X_train.sizes() << std::endl;
    std::cout << "Test set size: " << X_test.sizes() << std::endl;
    std::cout << "y_train shape: " << y_train.sizes() << std::endl;
    std::cout << "y_test shape: " << y_test.sizes() << std::endl;


    int epochs = 1000;
    float learningRate = 0.01;

    auto loss = torch::nn::BCELoss();
    auto model  = NNModel(X_train.size(1));

    auto optimizer = torch::optim::SGD(model->parameters(),learningRate);
    for (int epoch=0; epoch < epochs; ++epoch)
    {
        auto yPred = model->forward(X_train);
        auto loss1 = loss(yPred, y_train);
        optimizer.zero_grad();
        loss1.backward();
        optimizer.step();
        std::cout << "Epoch: " << epoch + 1
                 << ", Loss: " << loss1.item<double>() << "\n";

    }
    std::cout << "\n--- Test Set Evaluation ---\n";
    {
        torch::NoGradGuard no_grad;

        auto test_pred = model->forward(X_test);
        auto test_loss = loss(test_pred, y_test);

        // Accuracy: threshold at 0.5
        auto predicted = (test_pred >= 0.5).to(torch::kFloat32);
        auto correct   = predicted.eq(y_test).sum().item<int64_t>();
        double accuracy = static_cast<double>(correct) / X_test.size(0) * 100.0;

        std::cout << "Test Loss:     " << test_loss.item<double>() << "\n";
        std::cout << "Test Accuracy: " << accuracy << "%\n";
        std::cout << "Predictions:\n"  << test_pred << "\n";
        std::cout << "Ground truth:\n" << y_test    << "\n";
    }


}