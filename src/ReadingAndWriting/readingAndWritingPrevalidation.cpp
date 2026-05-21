//
// Created by moinshaikh on 3/12/26.
//#include<iostream>
#include"../external/third_party/doctest.hpp"
#include"csv.hpp"
#include<torch/torch.h>
#include<string>
#include<vector>
#include<map>

TEST_CASE("ReadingCSV_Optimized")
{
    csv::CSVFormat format;
    format.delimiter(',').no_header();

    csv::CSVReader reader("/home/moinshaikh/CLionProjects/LibtorchOpenCVTutorials/databases/data.csv", format);

    // 1. Initial Extraction
    std::vector<csv::CSVRow> rows(reader.begin(), reader.end());
    if (rows.empty()) return;

    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> csvData;

    // Separate Header from Data
    bool firstRow = true;
    for (auto& row : rows) {
        std::vector<std::string> currentRow;
        for (auto& cell : row)
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

    // --- STEP 2: PRE-VALIDATION (The "Cleaning" Phase) ---
    // Ensure every row is exactly the same length as the header
    size_t targetColCount = headers.size();
    for (auto& row : csvData)
    {
        if (row.size() < targetColCount)
        {
            row.resize(targetColCount, "0.0"); // Pad missing cells with "0.0"
        }
        else if (row.size() > targetColCount) {
            row.erase(row.begin() + targetColCount, row.end()); // Truncate extra cells
        }
    }

    // --- STEP 3: TENSOR GENERATION (The "Math" Phase) ---
    std::map<std::string, torch::Tensor> header_tensors;
    int num_rows = csvData.size();

    // Now we can iterate safely without bounds checks
    for (int col = 0; col < headers.size(); ++col) {
        std::string name = headers[col];

        // Skip non-numeric metadata (ID is col 0, Diagnosis is usually col 1)
        if (col == 0 || name == "diagnosis") continue;

        header_tensors[name] = torch::zeros({num_rows}, torch::kFloat32);

        for (int row = 0; row < num_rows; ++row) {
            try {
                // No 'if' or '.at()' needed here; validation guaranteed the size
                header_tensors[name][row] = std::stof(csvData[row][col]);
            } catch (...) {
                header_tensors[name][row] = 0.0f;
            }
        }
    }



    // --- STEP 4: ANALYSIS ---
    if (header_tensors.count("radius_mean") && header_tensors.count("texture_mean")) {
        auto r = header_tensors["radius_mean"];
        auto t = header_tensors["texture_mean"];

        // Pearson Correlation Logic
        auto r_c = r - r.mean();
        auto t_c = t - t.mean();
        auto corr = torch::sum(r_c * t_c) / (torch::sqrt(torch::sum(r_c * r_c)) * torch::sqrt(torch::sum(t_c * t_c)));

        std::cout << "Correlation: " << corr.item<float>() << std::endl;
    }
}