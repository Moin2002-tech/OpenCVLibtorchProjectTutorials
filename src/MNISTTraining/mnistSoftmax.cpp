//
// Created by moinshaikh on 7/24/26.
//
#include<iostream>
#include<torch/torch.h>
#include<torch/nn.h>
#include<Util.h>
#include "doctest.hpp"
#include<opencv2/opencv.hpp>

TEST_CASE("mnistSOftmax") {
    torch::Device device =  torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::manual_seed(777);
    if (device ==  torch::kCUDA) {
        print("cuda is available: ");
        torch::cuda::manual_seed_all(777);
    }

    //hyperParameters
    float learningRate = 0.001;
    int epochs = 30;
    int batchSize = 100;
    std::string root = "/home/moinshaikh/CLionProjects/LibtorchOpenCVTutorials/databases/mnist_data/";

    auto datasetsTrain = torch::data::datasets::MNIST(root,
        torch::data::datasets::MNIST::Mode::kTrain)
    .map(torch::data::transforms::Stack<>());
    const size_t total_batch =
    datasetsTrain.size().value() / batchSize;   // 60000 / 100 = 600

    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(datasetsTrain),
            torch::data::DataLoaderOptions()
                .batch_size(batchSize)
                .drop_last(true));

    torch::data::datasets::MNIST test_dataset(
    root,
    torch::data::datasets::MNIST::Mode::kTest);

    auto linear = torch::nn::Linear(784, 10);
    linear->to(device);

    auto criterion = torch::nn::CrossEntropyLoss();
    criterion->to(device);
    auto optimizer = torch::optim::Adam(linear->parameters(),learningRate);

    std::cout << "Starting training:\n"
              << "  device:      " << device        << "\n"
              << "  batch_size:  " << batchSize     << "\n"
              << "  epochs:      " << epochs<< "\n"
              << "  lr:          " << learningRate  << "\n"
              << "  total_batch: " << total_batch    << "\n\n";


    for (int64_t epoch = 0; epoch < epochs; ++epoch) {

        torch::Tensor avg_cost = torch::zeros({}, torch::kFloat32);

        for (auto& batch : *train_loader) {
            // X = X.view(-1, 28 * 28).to(device).to(kFloat) / 255
            auto X = batch.data.view({-1, 28 * 28}).to(device).to(torch::kFloat) / 255.0;
            // Y = Y.to(device)
            auto Y = batch.target.to(device);

            optimizer.zero_grad();

            auto hypothesis = linear->forward(X);
            auto cost = criterion->forward(hypothesis, Y);

            cost.backward();
            optimizer.step();

            avg_cost += cost.detach().cpu() / static_cast<double>(total_batch);
        }

        std::cout << "Epoch: " << std::setw(4) << std::setfill('0') << (epoch + 1)
                  << "  cost = " << std::fixed << std::setprecision(9)
                  << avg_cost.item<float>() << "\n";
    }

    std::cout << "\nLearning finished\n\n";

    // ========== Test the model using test sets ==========
    {
        torch::NoGradGuard no_grad;

        // Get full test data and labels as tensors
        auto X_test = test_dataset.images().to(device).to(torch::kFloat).view({-1, 28 * 28}) / 255.0;
        auto Y_test = test_dataset.targets().to(device);

        // Compute accuracy over entire test set
        auto prediction = linear->forward(X_test);
        auto correct_prediction = torch::argmax(prediction, 1).eq(Y_test);
        auto accuracy = correct_prediction.to(torch::kFloat).mean();
        std::cout << "Accuracy: " << accuracy.item<float>() << std::endl;

        // Get one random sample and predict
        int r = std::rand() % test_dataset.size().value();
        auto X_single = test_dataset.images().slice(0, r, r + 1)
                                .to(device).to(torch::kFloat).view({-1, 28 * 28}) / 255.0;
        auto Y_single = test_dataset.targets().slice(0, r, r + 1).to(device);

        std::cout << "Label: " << Y_single.item<int64_t>() << std::endl;
        auto single_prediction = linear->forward(X_single);
        auto predicted_class =  torch::argmax(single_prediction, 1).item<int64_t>();
        

                 //
        // Plot the image using OpenCV
        //
        //   1. Get the original image from test_dataset (<dtype>, [1, 28, 28])
        //   2. Clamp to [0,1] and scale to [0,255] if float; convert to uint8
        //   3. Convert to cv::Mat (grayscale, 28×28)
        //   4. Resize to 280×280 for visibility
        //   5. Overlay text: label (green) and prediction (blue if correct, red if wrong)
        //   6. Show with cv::imshow(), wait for a key press
        //
        // The raw image before any transforms (dtype may be uint8 or float)
        auto raw_image = test_dataset.images()
                             .slice(/*dim=*/0, /*start=*/r, /*end=*/r + 1)  // [1, 1, 28, 28]
                             .squeeze(0)                                      // remove batch dim → [1, 28, 28]
                             .squeeze(0)                                      // remove channel dim → [28, 28]
                             .cpu();                                          // ensure on CPU for data_ptr access

        // If the tensor is float, clamp to [0,1] and scale to [0,255] as uint8
        torch::Tensor display_tensor;
        if (raw_image.dtype() == torch::kFloat32 || raw_image.dtype() == torch::kFloat64) {
            display_tensor = (raw_image.clamp(0.0, 1.0) * 255.0).to(torch::kUInt8);
        } else if (raw_image.dtype() == torch::kUInt8) {
            display_tensor = raw_image;
        } else {
            // Fallback: attempt to cast
            display_tensor = raw_image.to(torch::kUInt8);
        }

        // Torch tensor (uint8, [28, 28]) → cv::Mat (grayscale, 28×28)
        cv::Mat img(28, 28, CV_8UC1, display_tensor.data_ptr<uint8_t>());

        // Resize to a visible size (280×280)
        cv::Mat img_display;
        cv::resize(img, img_display, cv::Size(280, 280), 0, 0, cv::INTER_NEAREST);

        // Convert grayscale to BGR so we can draw coloured text
        cv::Mat img_color;
        cv::cvtColor(img_display, img_color, cv::COLOR_GRAY2BGR);

        // Put text labels
        cv::putText(img_color,
                    "Label: " + std::to_string(Y_single.item<int64_t>()),
                    cv::Point(5, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0),   // green
                    2);

        // Prediction in blue if correct, red if wrong
        cv::Scalar pred_color = (predicted_class == Y_single.item<int64_t>())
                                    ? cv::Scalar(255, 0, 0)    // blue = correct
                                    : cv::Scalar(0, 0, 255);   // red  = wrong

        cv::putText(img_color,
                    "Pred: " + std::to_string(predicted_class),
                    cv::Point(5, 55),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    pred_color,
                    2);

        // Show the image
        cv::imshow("MNIST Test Sample", img_color);
        std::cout << "\nPress any key over the image window to continue...\n\n";
        cv::waitKey(0);
    }

    //visualizing it


}
