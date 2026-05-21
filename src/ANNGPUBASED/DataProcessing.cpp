//
// Created by moinshaikh on 5/1/26.
//

#include"DataProcessing.hpp"

#include<vector>
namespace DataProcessing
{
    DataProcessor::DataProcessor(std::string_view path, const csv::CSVFormat &format)
    {
        csv::CSVReader reader(path, format);
        rawData = std::vector<csv::CSVRow>(reader.begin(), reader.end());
    }

    void DataProcessor::extractCSVData()
    {
        // For no_header format, treat all rows as data
        // Create dummy headers for column indexing
        if (!rawData.empty()) {
            // Create headers based on first row column count
            size_t colCount = 0;
            for (auto &cell : rawData[0]) {
                colCount++;
            }
            
            // Create dummy headers (h0, h1, h2, ...)
            data.headers.clear();
            for (size_t i = 0; i < colCount; ++i) {
                data.headers.push_back("h" + std::to_string(i));
            }
            
            // Add all rows as data
            for (auto &row : rawData) {
                std::vector<std::string> currentRow;
                for (auto &cell : row) {
                    currentRow.push_back(cell.get<std::string>());
                }
                data.rows.push_back(currentRow);
            }
        }
    }

    void DataProcessor::PreValidation()
    {
        size_t targetColumnCount = data.headers.size();
        for (auto& row : data.rows)
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

    void DataProcessor::visualData(
        int grid_size,
        int original_img_size,
        int display_img_size,
        int spacing)
    {
        const int canvas_size = grid_size * display_img_size + (grid_size - 1) * spacing;
        // Create a white canvas
        // White background
        cv::Mat canvas = cv::Mat::zeros(canvas_size, canvas_size, CV_8UC1);
        canvas.setTo(255);  // White background


        // Process first 16 images
        for (int i = 0; i < 16 && i < rawData.size(); ++i)
        {
        const auto& row = rawData[i];

        // Extract label (first column)
        int label = row[0].get<int>();

        // Extract pixel values and reshape to 28x28
        cv::Mat img = cv::Mat::zeros(original_img_size, original_img_size, CV_8UC1);
        for (int j = 0; j < 784; ++j) {
        int pixel_value = row[j + 1].get<int>();
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
    }

    void DataProcessor::convertToTensor()
    {
        int rowCount = data.rows.size();
        int colCount = data.headers.size();

        
        if (rowCount == 0) {
            std::cout << "Error: No data rows found!" << std::endl;
            return;
        }
        
        if (!data.rows.empty())
            {
            for (int i = 0; i < std::min(5, (int)data.rows[0].size()); ++i)
            {
                std::cout << "'" << data.rows[0][i] << "' ";
            }
            std::cout << std::endl;
        }


        // 1. Pre-allocate continuous memory blocks
        // This stops C++ from re-allocating memory millions of times
        std::vector<float> all_pixels;
        all_pixels.reserve(rowCount * (colCount - 1));
        std::vector<float> all_labels;
        all_labels.reserve(rowCount);

        std::cout << "Parsing CSV data directly to flat buffers..." << std::endl;

        // 2. Single pass through the data matrix
        for (int row = 0; row < rowCount; ++row) {
            for (int col = 0; col < colCount; ++col) {
                const std::string& cell_value = data.rows[row][col]; //csvData.second[row][col];
                float val = 0.0f;



                // Fast parse with a lightweight fallback
                if (!cell_value.empty()) {
                    try {
                        val = std::stof(cell_value);
                    } catch (const std::exception& e) {
                        std::cout << "Error parsing '" << cell_value << "' at row=" << row << ", col=" << col << ": " << e.what() << std::endl;
                        val = 0.0f; // Handle garbage data safely
                    } catch (...) {
                        std::cout << "Unknown error parsing '" << cell_value << "' at row=" << row << ", col=" << col << std::endl;
                        val = 0.0f; // Handle garbage data safely
                    }
                }

                // 3. Route labels to one buffer, pixels to the other
                if (col == 0)

                    {
                    all_labels.push_back(val);
                } else
                    {
                    all_pixels.push_back(val);
                }
            }
        }

        tensorData.X = torch::from_blob(all_pixels.data(), {rowCount, colCount - 1}, torch::kFloat32).clone();
        tensorData.Y = torch::from_blob(all_labels.data(), {rowCount}, torch::kFloat32).clone();
        std::cout << "X shape: " << tensorData.X.sizes() << std::endl;
        std::cout << "y shape: " << tensorData.X.sizes() << std::endl;

    }

    void DataProcessor::SplitData()
    {
        int totalSample = tensorData.X.size(0);
        int test_size     = static_cast<int>(totalSample * 0.2);
        int train_size    = totalSample - test_size;

        auto indices       = torch::randperm(totalSample);
        auto train_indices = indices.slice(0, 0, train_size);
        auto test_indices  = indices.slice(0, train_size, totalSample);

        auto X_train = tensorData.X.index_select(0, train_indices);
        auto X_test  = tensorData.X.index_select(0, test_indices);
        auto y_train = tensorData.Y.index_select(0, train_indices).to(torch::kInt64);
        auto y_test  = tensorData.Y.index_select(0, test_indices).to(torch::kInt64);
        // scaling the feautures
        X_train = X_train/255.0;
        X_test = X_test/255.0;

        datasets.X_train = X_train;
        datasets.Y_train = y_train;
        datasets.X_test = X_test;
        datasets.Y_test = y_test;

        datasets.X_train = datasets.X_train / 255.0;
        datasets.X_test = datasets.X_test / 255.0;
    }

}
