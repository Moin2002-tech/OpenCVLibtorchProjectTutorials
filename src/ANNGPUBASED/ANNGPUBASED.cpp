//
// Created by moinshaikh on 5/2/26.
//
#include"../../external/third_party/doctest.hpp"

#include"CNNImple.hpp"
#include"CustomDatasets.hpp"
#include"DataProcessing.hpp"

using namespace DataProcessing;

TEST_CASE("ANNGPUBASED")
{

            csv::CSVFormat format;
            format.delimiter(',');
            std::cout << "Creating DataProcessor..." << std::endl;
            DataProcessor process("/home/moinshaikh/CLionProjects/LibtorchOpenCVTutorials/databases/fashion-mnist_train.csv",format);
            std::cout << "DataProcessor created successfully" << std::endl;
            
            std::cout << "Calling extractCSVData..." << std::endl;
            process.extractCSVData();
            std::cout << "extractCSVData completed successfully" << std::endl;
            
            std::cout << "Calling PreValidation..." << std::endl;
            process.PreValidation();
            std::cout << "PreValidation completed successfully" << std::endl;
            
            std::cout << "Calling convertToTensor..." << std::endl;
            process.convertToTensor();
            std::cout << "convertToTensor completed successfully" << std::endl;
            
            std::cout << "Calling visualData..." << std::endl;
            //process.visualData();
            std::cout << "visualData completed successfully" << std::endl;

            std::cout<<"converting To tensor";
            process.convertToTensor();

            std::cout<<"splitting data";
            process.SplitData();

            auto datasets = process.getDatasets();
            std::cout<<"splitting completed successfully"<<std::endl;

    torch::manual_seed(42);
    torch::Device device = torch::kCPU;

    auto train_dataset = CustomDatasets(datasets.X_train, datasets.Y_train).map(torch::data::transforms::Stack<>());
    auto test_dataset  = CustomDatasets(datasets.X_test, datasets.Y_test).map(torch::data::transforms::Stack<>());

    int batchSize = 256;

    // 2. Use std::move() to transfer ownership to the DataLoader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(batchSize).workers(4)
    );

    // Use SequentialSampler for test (no shuffling needed)
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset),
        torch::data::DataLoaderOptions().batch_size(batchSize).workers(4)
    );

    std::cout<<"Data loading completed successfully"<<std::endl;


    float learningRate = 0.01;
    int epochs = 100;

    auto model = CNN(1);
    model->to(device);
    auto criterion = torch::nn::CrossEntropyLoss();
    auto weightDecay = 1e-4;
    auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(learningRate).weight_decay(weightDecay));

    for (int epoch = 0 ; epoch< epochs; ++epoch)
    {
        // Evaluate every 5 epochs to save time
        if (epoch % 5 == 0) {
            model->eval();
            // Add quick evaluation here if needed
            model->train();
        }
        model->train();
        for (auto batch : *train_loader)
        {
            auto batchFeature = batch.data.to(device);
            auto batchLabel   = batch.target.to(device);

            optimizer.zero_grad();
            auto yPred = model->forward(batchFeature);
            auto loss  = criterion(yPred, batchLabel);
            loss.backward();
            optimizer.step();

             //std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<double>() << "\n";
        }
    }

    model->eval();
    int total = 0;
    int correct = 0;

    torch::NoGradGuard no_grad;

    for (auto batch : *test_loader)
    {
        auto batch_features = batch.data.to(device);
        auto batch_labels = batch.target.to(device);

        auto outputs = model->forward(batch_features);
        auto predictions = torch::max(outputs, 1);
        auto predicted = std::get<1>(predictions);

        total += batch_labels.size(0);
        correct += (predicted == batch_labels).sum().item<int>();
    }

    float accuracy = static_cast<float>(correct) / total;
    std::cout << "Test Accuracy: " << accuracy * 100.0f << "%" << std::endl;

}