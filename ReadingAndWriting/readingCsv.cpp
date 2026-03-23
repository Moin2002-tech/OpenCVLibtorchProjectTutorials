//
// Created by moinshaikh on 3/12/26.
//

#include<iostream>
#include"../external/third_party/doctest.hpp"
#include"csv.hpp"
#include<torch/torch.h>
#include<string>
#include<vector>
#include<map>

template<class T>
void print(const T&t)
{
    std::cout<<t;
}

TEST_CASE("ReadingCSV")
{
    csv::CSVFormat format;
    format.delimiter(',').no_header();

    csv::CSVReader reader("/home/moinshaikh/CLionProjects/LibtorchOpenCVTutorials/databases/data.csv",format);
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> csvData;
    int count = 0;
    std::vector<csv::CSVRow> rows(reader.begin(),reader.end());

    int totalElement = 0;
    for (const auto& row : rows)
    {
        totalElement += row.size();
    }

    for (auto & row : rows)
    {
        std::vector<std::string> currentRow;
        for (int col = 0; col< row.size();++col)
        {
            try
            {
                currentRow.push_back(row[col].get<std::string>());
            }
            catch (...)
            {
                break;
            }
        }
        if (count == 0)
        {
            headers = currentRow;
            print("headers found:\t");
            print(headers.size());
        }
        else
        {
            csvData.push_back(currentRow);
        }
        count++;
    }

    // Create individual tensors for each header
    std::map<std::string, torch::Tensor> header_tensors;
    int num_rows = csvData.size();
    
    // Debug info
    std::cout << "Headers size: " << headers.size() << std::endl;
    std::cout << "CSV Data size: " << csvData.size() << std::endl;
    if (!csvData.empty()) {
        std::cout << "First row size: " << csvData[0].size() << std::endl;
    }
    
    // Create tensors for each column (skip ID and diagnosis columns)
    for (int col = 1; col < headers.size(); ++col)
        {
        std::string header_name = headers[col];
        
        // Skip non-numeric columns
        if (header_name == "diagnosis") {
            std::cout << "Skipping non-numeric column: " << header_name << std::endl;
            continue;
        }
        
        std::cout << "Processing column: " << header_name << " (index " << col << ")" << std::endl;
        
        header_tensors[header_name] = torch::zeros({num_rows}, torch::kFloat32);
        
        // Fill the tensor with data for this column
        for (int row = 0; row < csvData.size(); ++row) {
            // Bounds check
            if (col >= csvData[row].size()) {
                std::cout << "Column index " << col << " out of bounds for row " << row << std::endl;
                header_tensors[header_name][row] = 0.0f;
                continue;
            }
            
            try {
                std::string cell_value = csvData[row][col];
                // Check if the string is not empty and contains valid numeric data
                if (!cell_value.empty() && 
                    (isdigit(cell_value[0]) || cell_value[0] == '-' || cell_value[0] == '.')) {
                    float value = std::stof(cell_value);
                    header_tensors[header_name][row] = value;
                } else {
                    header_tensors[header_name][row] = 0.0f;
                }
            } catch (...) {
                header_tensors[header_name][row] = 0.0f; // Handle conversion errors
            }
        }
    }
    
    // Print info for each header tensor
    std::cout << "Created " << header_tensors.size() << " header tensors:" << std::endl;
    for (const auto& [header, tensor] : header_tensors) {
        std::cout << header << ": shape " << tensor.sizes() 
                  << ", mean " << torch::mean(tensor).item<float>() << std::endl;
    }
    
    // Example operations with specific columns
    if (header_tensors.find("radius_mean") != header_tensors.end() && 
        header_tensors.find("texture_mean") != header_tensors.end()) {
        
        auto radius_mean = header_tensors["radius_mean"];
        auto texture_mean = header_tensors["texture_mean"];
        
        std::cout << "\nRadius mean stats:" << std::endl;
        std::cout << "Min: " << torch::min(radius_mean).item<float>() << std::endl;
        std::cout << "Max: " << torch::max(radius_mean).item<float>() << std::endl;
        std::cout << "Std: " << torch::std(radius_mean).item<float>() << std::endl;
        
        // Correlation between radius and texture
        auto radius_centered = radius_mean - torch::mean(radius_mean);
        auto texture_centered = texture_mean - torch::mean(texture_mean);
        auto correlation = torch::sum(radius_centered * texture_centered) / 
                          (torch::sqrt(torch::sum(radius_centered * radius_centered)) * 
                           torch::sqrt(torch::sum(texture_centered * texture_centered)));
        std::cout << "Correlation radius vs texture: " << correlation.item<float>() << std::endl;
    }
}
