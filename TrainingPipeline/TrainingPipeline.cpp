//
// Created by moinshaikh on 3/12/26.
//

#include<map>
#include<iostream>
#include<string>
#include<vector>
#include<torch/torch.h>
#include"../external/third_party/doctest.hpp"
#include"csv.hpp"


void preValidation(std::vector<std::string> &headers, std::vector<std::vector<std::string>> &csvData)
{
    size_t targetColumnCount = headers.size();
    for (auto& row : csvData)
    {
        if (row.size() < targetColumnCount)
        {
            row.resize(targetColumnCount, "0.0");
        }
        else if (row.size() > targetColumnCount)
        {
            row.erase(row.begin() + targetColumnCount, row.end()); // Truncate extra cells
        }
    }

}
class MySimpleNNImpl : public torch::nn::Module {
public:
    torch::Tensor weights;
    torch::Tensor bias;

    MySimpleNNImpl(int n_features)
    {
        weights = register_parameter("weights",
            torch::rand({n_features, 1}, torch::dtype(torch::kFloat64)));
        bias = register_parameter("bias",
            torch::zeros({1}, torch::dtype(torch::kFloat64)));
    }

    torch::Tensor forward(torch::Tensor X) {
        auto z = torch::matmul(X, weights) + bias;
        return torch::sigmoid(z);
    }

    torch::Tensor loss_function(torch::Tensor y_pred, torch::Tensor y) {
        const double epsilon = 1e-7;
        y_pred = torch::clamp(y_pred, epsilon, 1.0 - epsilon);
        auto loss = -(y * torch::log(y_pred) +
                     (1 - y) * torch::log(1 - y_pred)).mean();
        return loss;
    }
};
TORCH_MODULE(MySimpleNN);

TEST_CASE("trainingPipeLine")
{
    std::string path = "/home/moinshaikh/CLionProjects/LibtorchOpenCVTutorials/databases/data.csv";
    csv::CSVFormat format;
    format.delimiter(',').no_header();

    csv::CSVReader reader(path,format);
    std::vector<csv::CSVRow> rows(reader.begin(),reader.end());
    if (rows.empty()) return;

    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> csvData;

    bool firstRow = true;
    for (auto & row : rows)
    {
        std::vector<std::string> currentRow;
        for (auto &cell : row )
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

    //Pre validation
    size_t targetColCount = headers.size();
    for (auto& row : csvData)
    {
        if (row.size() < targetColCount)
        {
            row.resize(targetColCount, "0.0");
        }
        else if (row.size() > targetColCount)
        {
            row.erase(row.begin() + targetColCount, row.end()); // Truncate extra cells
        }
    }

    // --- STEP 3: DIRECT TENSOR CREATION AND ENCODING ---
    std::map<std::string, torch::Tensor> header_tensors;
    int num_rows = csvData.size();

    // Create tensors for feature columns (skip ID as it's not a feature)
    for (int col = 0; col < headers.size(); ++col) {
        std::string name = headers[col];
        
        // Skip ID column - not used as a feature
        if (name == "id") continue;
        
        // Create tensor for this column
        header_tensors[name] = torch::zeros({num_rows}, torch::kFloat64);
        
        // Encode directly into tensor based on column type
        for (int row = 0; row < num_rows; ++row)
            {
            std::string cell_value = csvData[row][col];
            
            if (name == "diagnosis")
            {
                // Label encoding: M->1, B->0
                header_tensors[name][row] = (cell_value == "M") ? 1.0f : 0.0f;
            }
            else
            {
                // Numeric columns
                try {
                    header_tensors[name][row] = std::stod(cell_value);
                } catch (...) {
                    header_tensors[name][row] = 0.0f;
                }
            }
        }
    }

    // --- STEP 4: CREATE FEATURE MATRIX AND TARGET VECTOR ---
    // Stack all feature tensors (exclude diagnosis)
    std::vector<torch::Tensor> feature_tensors;
    for (auto& [name, tensor] : header_tensors)
    {
        if (name != "diagnosis")
        {
            feature_tensors.push_back(tensor);
        }
    }
    
    // Create feature matrix X and target y
    auto X = torch::stack(feature_tensors, 1); // Stack along dimension 1 (columns)
    auto y = header_tensors["diagnosis"];

    std::cout << "Feature matrix shape: " << X.sizes() << std::endl;
    std::cout << "Target vector shape: " << y.sizes() << std::endl;

    // Normalize features (zero mean, unit variance)
    auto X_mean = X.mean(0, true);
    auto X_std  = X.std(0, true);
    X_std = torch::clamp(X_std, 1e-7);
    X = (X - X_mean) / X_std;

    std::cout << "Feature matrix shape: " << X.sizes() << "\n";
    std::cout << "Target vector shape: "  << y.sizes() << "\n";

    // --- STEP 5: TRAIN/TEST SPLIT ---
    int total_samples = X.size(0);
    int test_size     = static_cast<int>(total_samples * 0.2);
    int train_size    = total_samples - test_size;

    auto indices       = torch::randperm(total_samples);
    auto train_indices = indices.slice(0, 0, train_size);
    auto test_indices  = indices.slice(0, train_size, total_samples);

    auto X_train = X.index_select(0, train_indices);
    auto X_test  = X.index_select(0, test_indices);
    auto y_train = y.index_select(0, train_indices).unsqueeze(1);
    auto y_test  = y.index_select(0, test_indices).unsqueeze(1);

    std::cout << "Training set size: " << X_train.sizes() << std::endl;
    std::cout << "Test set size: " << X_test.sizes() << std::endl;
    std::cout << "y_train shape: " << y_train.sizes() << std::endl;
    std::cout << "y_test shape: " << y_test.sizes() << std::endl;


    int epochs = 1000;
    float learningRate = 0.01;

    MySimpleNN model(X_train.size(1));

    // ------- Training Loop -------
    for (int epoch = 0; epoch < epochs; ++epoch) {

        // Forward pass
        auto y_pred = model->forward(X_train);

        // Loss
        auto loss = model->loss_function(y_pred, y_train);

        // Backward pass
        loss.backward();

        // Parameter update (manual, no optimizer)
        {
            torch::NoGradGuard no_grad;
            model->weights -= learningRate * model->weights.grad();
            model->bias    -= learningRate * model->bias.grad();
        }

        // Zero gradients
        model->weights.grad().zero_();
        model->bias.grad().zero_();

        std::cout << "Epoch: " << epoch + 1
                  << ", Loss: " << loss.item<double>() << "\n";
    }

    // ------- Evaluation on Test Set -------
    std::cout << "\n--- Test Set Evaluation ---\n";
    {
        torch::NoGradGuard no_grad;

        auto test_pred = model->forward(X_test);
        auto test_loss = model->loss_function(test_pred, y_test);

        // Accuracy: threshold at 0.5
        auto predicted = (test_pred >= 0.5).to(torch::kFloat64);
        auto correct   = predicted.eq(y_test).sum().item<int64_t>();
        double accuracy = static_cast<double>(correct) / X_test.size(0) * 100.0;

        std::cout << "Test Loss:     " << test_loss.item<double>() << "\n";
        std::cout << "Test Accuracy: " << accuracy << "%\n";
        std::cout << "Predictions:\n"  << test_pred << "\n";
        std::cout << "Ground truth:\n" << y_test    << "\n";
    }






}