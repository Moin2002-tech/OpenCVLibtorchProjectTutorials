//
// Created by moinshaikh on 3/19/26.
//


#include<iostream>
#include<torch/torch.h>
#include<vector>
#include<csv.hpp>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include<algorithm>
#include"../external/third_party/doctest.hpp"

template<typename T>
void print(const T& t)
{
    std::cout<<t<<"\n";
}


std::vector<csv::CSVRow> rawData(const std::string &path,const csv::CSVFormat &format)
{
        csv::CSVReader reader(path, format);
        std::vector<csv::CSVRow> rows(reader.begin(), reader.end());
        return rows;
}

std::pair<std::vector<std::string>,std::vector<std::vector<std::string>>> extractData(const std::vector<csv::CSVRow> &rows)
{
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> csvData;

    bool firstRow = true;
    for(auto &row:rows)
    {
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

    return std::make_pair(headers, csvData);
}

void preValidation(std::pair<std::vector<std::string>,std::vector<std::vector<std::string>>>& csvData)
{
    size_t targetColumnCount = csvData.first.size();
        for (auto& row : csvData.second)
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

class CustomDatasets: public torch::data::Dataset<CustomDatasets>
{
private:
    torch::Tensor features;
    torch::Tensor labels;
public:
    CustomDatasets(torch::Tensor features, torch::Tensor labels) : features(features), labels(labels)
    {

    }

    // Now this will correctly link with your Dataset definition
    torch::data::Example<> get(size_t index) override
    {
        return {features[index], labels[index]};
    }

    torch::optional<size_t> size() const override
    {
        return features.size(0);
    }

};


class ANNImpl : public torch::nn::Module
{
public:
    torch::nn::Sequential model{nullptr};

    ANNImpl(int numFeatures)
    {
        model =torch::nn::Sequential(torch::nn::Linear(numFeatures,128),
            torch::nn::ReLU(),
            torch::nn::Linear(128,64),
            torch::nn::ReLU(),
            torch::nn::Linear(64,10)
            );

        register_module("model",model);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return model->forward(x);
    }
}; TORCH_MODULE(ANN);

TEST_CASE("ANNFashionMNIST")
{
    std::string path = "/home/moinshaikh/CLionProjects/LibtorchOpenCVTutorials/databases/fmnist_small.csv";
    csv::CSVFormat format;
    format.delimiter(',').no_header();
   // csv::CSVReader reader(path, format);

    //std::vector<csv::CSVRow> rows(reader.begin(), reader.end());
    
    // Create a 4x4 grid to display first 16 images
    const int grid_size = 4;
    const int original_img_size = 28;  // MNIST images are 28x28
    const int display_img_size = 100;  // Scale up for better visibility
    const int spacing = 15;   // Space between images
    const int canvas_size = grid_size * display_img_size + (grid_size - 1) * spacing;
    
    // Create a white canvas
    cv::Mat canvas = cv::Mat::zeros(canvas_size, canvas_size, CV_8UC1);
    canvas.setTo(255);  // White background

    /*
    // Process first 16 images
    for (int i = 0; i < 16 && i < rows.size(); ++i)
        {
        const auto& row = rows[i];
        
        // Extract label (first column)
        int label = row["label"].get<int>();
        
        // Extract pixel values and reshape to 28x28
        cv::Mat img = cv::Mat::zeros(original_img_size, original_img_size, CV_8UC1);
        for (int j = 0; j < 784; ++j) {
            std::string col_name = "pixel" + std::to_string(j + 1);
            int pixel_value = row[col_name].get<int>();
            img.at<uchar>(j / original_img_size, j % original_img_size) = static_cast<uchar>(pixel_value);
        }
        
        // Resize image for better visibility
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(display_img_size, display_img_size), 0, 0, cv::INTER_NEAREST);
        
        // Calculate position in the grid
        int row_pos = i / grid_size;
        int col_pos = i % grid_size;
        int x = col_pos * (display_img_size + spacing);
        int y = row_pos * (display_img_size + spacing);
        
        // Copy resized image to canvas
        cv::Rect roi(x, y, display_img_size, display_img_size);
        resized_img.copyTo(canvas(roi));
        
        // Add label text above the image
        std::string label_text = "Label: " + std::to_string(label);
        cv::putText(canvas, label_text, 
                   cv::Point(x + 5, y - 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 
                   0.4, 
                   cv::Scalar(0),  // Black text
                   1);
    }
    
    // Display the grid
    cv::imshow("First 16 Images - Fashion MNIST", canvas);
    cv::waitKey(0);  // Wait for key press
    cv::destroyAllWindows();

    */

    auto rows = rawData(path,format);
    auto csvData = extractData(rows);
    preValidation(csvData);


    std::map<std::string,torch::Tensor> headersTensors;
    int rowCount =  csvData.second.size();

    for (int col =0 ; col < csvData.first.size(); ++col)
    {
        std::string name = csvData.first[col];
        if (name.empty())
            continue;

        headersTensors[name] = torch::zeros({rowCount},torch::kFloat32);
        for (int row = 0; row < rowCount; ++row)
        {
            std::string cell_value = csvData.second[row][col];
            if (name == "label")
            {
                // Fashion MNIST labels are already numeric (0-9)
                try
                {
                    headersTensors[name][row] = std::stof(cell_value);
                }
                catch (...)
                {
                    headersTensors[name][row] = 0.0f;
                }
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

    torch::manual_seed(44);
    std::vector<torch::Tensor> features;
    for (auto & [name, tensor]: headersTensors)
    {
        if (name != "label")
        {features.push_back(tensor);}
    }
    auto X = torch::stack(features,1);
    auto Y = headersTensors["label"];

    std::cout << "X shape: " << X.sizes() << std::endl;
    std::cout << "y shape: " << Y.sizes() << std::endl;

    //NORMALISE
    // Normalize features (zero mean, unit variance)

    std::cout << "Feature matrix shape: " << X.sizes() << "\n";
    std::cout << "Target vector shape: "  << Y.sizes() << "\n";

    int total_samples = X.size(0);
    int test_size     = static_cast<int>(total_samples * 0.2);
    int train_size    = total_samples - test_size;

    auto indices       = torch::randperm(total_samples);
    auto train_indices = indices.slice(0, 0, train_size);
    auto test_indices  = indices.slice(0, train_size, total_samples);

    auto X_train = X.index_select(0, train_indices);
    auto X_test  = X.index_select(0, test_indices);
    auto y_train = Y.index_select(0, train_indices).to(torch::kInt64);
    auto y_test  = Y.index_select(0, test_indices).to(torch::kInt64);
    // scaling the feautures
    X_train = X_train/255.0;
    X_test = X_test/255.0;
    // 1. Map the stack transform right when you create the dataset
    auto train_dataset = CustomDatasets(X_train, y_train).map(torch::data::transforms::Stack<>());
    auto test_dataset  = CustomDatasets(X_test, y_test).map(torch::data::transforms::Stack<>());

    std::cout<<"Training set size: "<<*train_dataset.size()<<"\n";
    std::cout<<"Test set size: "<<*test_dataset.size()<<"\n";

    int batchSize = 32;

    // 2. Use std::move() to transfer ownership to the DataLoader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(batchSize)
    );

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(test_dataset),
        torch::data::DataLoaderOptions().batch_size(batchSize)
    );


    int epochs = 100;
    float learningRate = 0.01;


    auto model = ANN(X_train.size(1));
    model->to(torch::kCUDA);

    //loss function
    auto criterion = torch::nn::CrossEntropyLoss();

    auto optimizer = torch::optim::SGD(model->parameters(), learningRate);

    //train loop
    for (int epoch = 0 ; epoch< epochs; ++epoch)
    {
        model->train();
        for (auto batch : *train_loader)
        {
            auto batchFeature = batch.data.to(torch::kCUDA);
            auto batchLabel   = batch.target.to(torch::kCUDA);

            optimizer.zero_grad();
            auto yPred = model->forward(batchFeature);
            auto loss  = criterion(yPred, batchLabel);
            loss.backward();
            optimizer.step();

            std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<double>() << "\n";
        }
    }
    //evaluation:
    model->eval();
    
    int total = 0;
    int correct = 0;
    
    torch::NoGradGuard no_grad;
    
    for (auto batch : *test_loader)
    {
        auto batch_features = batch.data.to(torch::kCUDA);
        auto batch_labels = batch.target.to(torch::kCUDA);
        
        auto outputs = model->forward(batch_features);
        auto predictions = torch::max(outputs, 1);
        auto predicted = std::get<1>(predictions);
        
        total += batch_labels.size(0);
        correct += (predicted == batch_labels).sum().item<int>();
    }
    
    float accuracy = static_cast<float>(correct) / total;
    std::cout << "Test Accuracy: " << accuracy * 100.0f << "%" << std::endl;


}