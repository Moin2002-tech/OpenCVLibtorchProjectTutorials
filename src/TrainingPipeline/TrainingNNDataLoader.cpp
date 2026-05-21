//
// Created by moinshaikh on 3/14/26.
//


#include<torch/torch.h>
#include<torch/nn.h>
#include<torch/data/dataloader.h>
#include<torch/data/datasets.h>
#include<csv.hpp>
#include"../external/third_party/doctest.hpp"

// Corrected inheritance line
class CustomDatasets : public torch::data::datasets::Dataset<CustomDatasets>
{
private:
    torch::Tensor features;
    torch::Tensor labels;

public:
    CustomDatasets(torch::Tensor features, torch::Tensor labels)
        : features(features), labels(labels) {}

    // Now this will correctly link with your Dataset definition
    torch::data::Example<> get(size_t index) override {
        return {features[index], labels[index]};
    }

    torch::optional<size_t> size() const override {
        return features.size(0);
    }
};

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

TEST_CASE("TrainingNNDataLoader")
{
    std::string path = "/home/moinshaikh/CLionProjects/LibtorchOpenCVTutorials/databases/data.csv";
    csv::CSVFormat format;
    format.delimiter(',').no_header();
    csv::CSVReader reader(path, format);
    std::vector<csv::CSVRow> rows(reader.begin(),reader.end());

    std::vector<std::string>  headers;
    std::vector<std::vector<std::string>> csvData;

    bool firstRow = true;
    for (auto &row : rows) {
        std::vector<std::string> currentRow;
        for (auto &cell : row)
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

    // 1. Map the stack transform right when you create the dataset
    auto train_dataset = CustomDatasets(X_train, y_train).map(torch::data::transforms::Stack<>());
    auto test_dataset  = CustomDatasets(X_test, y_test).map(torch::data::transforms::Stack<>());

    std::cout<<"Training set size: "<<*train_dataset.size()<<"\n";
    std::cout<<"Test set size: "<<*test_dataset.size()<<"\n";

    int batchsize = 32;

    // 2. Use std::move() to transfer ownership to the DataLoader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(batchsize)
    );

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(test_dataset),
        torch::data::DataLoaderOptions().batch_size(batchsize)
    );

    float learningRate = 0.01;
    int epochs = 25;

    auto model = NNModel(X_train.size(1));
    auto optimizer = torch::optim::SGD(model->parameters(), learningRate);
    auto lossFunction = torch::nn::BCELoss();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        model->train();
        for (auto batch : *train_loader) {
            auto batchFeature = batch.data;
            auto batchLabel   = batch.target;

            optimizer.zero_grad();
            auto yPred = model->forward(batchFeature);
            auto loss  = lossFunction(yPred, batchLabel);
            loss.backward();
            optimizer.step();

            std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<double>() << "\n";
        }
    }
    std::cout << "\n--- Test Set Evaluation ---\n";
    {
        torch::NoGradGuard no_grad;
        model->eval();

        double total_loss = 0.0;
        int64_t correct_preds = 0;
        int64_t total_samples = 0;
        int64_t num_batches = 0; // Backup counter for the average

        for (auto& batch : *test_loader) {
            auto data    = batch.data;
            auto targets = batch.target;

            auto output = model->forward(data);  // already has sigmoid inside


            auto loss_tensor = torch::binary_cross_entropy(output, targets.to(torch::kFloat32));
            total_loss += loss_tensor.item<double>();

            auto predicted = (output >= 0.5).to(torch::kInt64);
            correct_preds += predicted.reshape_as(targets).eq(targets).sum().item<int64_t>();
            total_samples += data.size(0);
            num_batches++;
        }

        // FIX: value_or() needs an argument, e.g., 1 to avoid division by zero
        // Or better yet, use the manual num_batches we just calculated
        double avg_loss = total_loss / (num_batches > 0 ? num_batches : 1);
        double accuracy = (total_samples > 0) ? (static_cast<double>(correct_preds) / total_samples * 100.0) : 0.0;

        std::cout << "Test Loss:     " << avg_loss << "\n";
        std::cout << "Test Accuracy: " << accuracy << "%\n";
    }

}










