//
// Created by moinshaikh on 6/28/26.
//

#include <doctest.hpp>
#include <Util.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <random>

TEST_CASE("MNIST_training_and_test") {

    // --- Device setup ---
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::manual_seed(777);
    if (device == torch::kCUDA) {
        torch::cuda::manual_seed(777);
    }

    // --- Hyper-parameters ---
    constexpr int64_t kBatchSize      = 100;
    constexpr int64_t kTrainingEpochs = 15;
    constexpr double  kLearningRate   = 0.1;

    const std::string kDataRoot =
        "/home/moinshaikh/CLionProjects/LibtorchOpenCVTutorials/databases/mnist_data/";

    // ================================================================
    // Training dataset & DataLoader
    // ================================================================
    auto train_dataset =
        torch::data::datasets::MNIST(kDataRoot,
                                     torch::data::datasets::MNIST::Mode::kTrain)
        .map(torch::data::transforms::Normalize<>(0, 1.0 / 255.0))
        .map(torch::data::transforms::Stack<>());

    const size_t total_batch =
        train_dataset.size().value() / kBatchSize;   // 60000 / 100 = 600

    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset),
            torch::data::DataLoaderOptions()
                .batch_size(kBatchSize)
                .drop_last(true));


    torch::data::datasets::MNIST test_dataset(
        kDataRoot,
        torch::data::datasets::MNIST::Mode::kTest);


    torch::nn::Linear linear(784, 10);
    linear->to(device);

    auto criterion = torch::nn::CrossEntropyLoss();

    torch::optim::SGD optimizer(linear->parameters(),
                                torch::optim::SGDOptions(kLearningRate));

    std::cout << "Starting training:\n"
              << "  device:      " << device        << "\n"
              << "  batch_size:  " << kBatchSize     << "\n"
              << "  epochs:      " << kTrainingEpochs<< "\n"
              << "  lr:          " << kLearningRate  << "\n"
              << "  total_batch: " << total_batch    << "\n\n";


    for (int64_t epoch = 0; epoch < kTrainingEpochs; ++epoch) {

        torch::Tensor avg_cost = torch::zeros({}, torch::kFloat32);

        for (auto& batch : *train_loader) {
            // X = X.view(-1, 28 * 28).to(device)
            auto X = batch.data.view({-1, 28 * 28}).to(device);
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


    {
        torch::NoGradGuard no_grad;   // with torch.no_grad()


        auto X_test = test_dataset.images()
                          .to(torch::kFloat32)           // uint8 → float32
                          .view({-1, 28 * 28})           // [10000, 784]
                          .to(device);


        auto Y_test = test_dataset.targets()
                          .to(device);                   // [10000]


        auto prediction = linear->forward(X_test);       // [10000, 10]


        auto correct_prediction = torch::argmax(prediction, /*dim=*/1)
                                      .eq(Y_test);       // [10000] boolean

        // accuracy = correct_prediction.float().mean()
        auto accuracy = correct_prediction
                            .to(torch::kFloat32)
                            .mean();

        std::cout << "Accuracy: " << accuracy.item<float>() << "\n";


        auto total_test_samples = test_dataset.images().size(0);
        std::mt19937 gen(42);                        // fixed seed for reproducibility
        std::uniform_int_distribution<int64_t> dist(0, total_test_samples - 1);
        int64_t r = dist(gen);                       // random index


        auto X_single = test_dataset.images()
                            .slice(/*dim=*/0, /*start=*/r, /*end=*/r + 1)
                            .to(torch::kFloat32)
                            .view({-1, 28 * 28})
                            .to(device);


        auto Y_single = test_dataset.targets()
                            .slice(/*dim=*/0, /*start=*/r, /*end=*/r + 1)
                            .to(device);

        std::cout << "Label: " << Y_single.item<int64_t>() << "\n";

        // single_prediction = linear(X_single_data)
        auto single_prediction = linear->forward(X_single);
        auto predicted_class = torch::argmax(single_prediction, /*dim=*/1).item<int64_t>();
        std::cout << "Prediction: " << predicted_class << "\n";

        // ================================================================
        // Plot the image using OpenCV
        //
        //   1. Get the original image from test_dataset (<dtype>, [1, 28, 28])
        //   2. Clamp to [0,1] and scale to [0,255] if float; convert to uint8
        //   3. Convert to cv::Mat (grayscale, 28×28)
        //   4. Resize to 280×280 for visibility
        //   5. Overlay text: label (green) and prediction (blue if correct, red if wrong)
        //   6. Show with cv::imshow(), wait for a key press
        // ================================================================
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



}