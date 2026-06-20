//
// Created by moinshaikh on 5/25/26.
//
#pragma once
#ifndef LIBTORCHOPENCVTUTORIALS_UTIL_H
#define LIBTORCHOPENCVTUTORIALS_UTIL_H

#include <iostream>
#include <torch/torch.h>
#include<csv.hpp>

template <typename T>
void print(const std::string& label, const T& t) {
    std::cout << label << ":\n" << t << "\n\n";
}

inline void print(const std::string &label) {
    std::cout<< label << "\n\n";
}

// Overload for torch::Tensor::max(dim) which returns (values, indices) tuple
inline void print(const std::string& label, const std::tuple<torch::Tensor, torch::Tensor>& t) {
    std::cout << label << ":\n";
    std::cout << "  values:  " << std::get<0>(t) << "\n";
    std::cout << "  indices: " << std::get<1>(t) << "\n\n";
}

struct datasets {
    torch::Tensor xTrain;
    torch::Tensor yTrain;
};

inline datasets convertIntoTensor(std::string_view path, datasets& data)
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

#endif //LIBTORCHOPENCVTUTORIALS_UTIL_H
