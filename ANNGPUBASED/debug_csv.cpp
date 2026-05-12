#include"../external/third_party/doctest.hpp"
#include<csv.hpp>
/*
TEST_CASE("Debug CSV")
{
    std::cout << "Testing CSV reading..." << std::endl;
    
    try {
        csv::CSVFormat format;
        format.delimiter(',').no_header();
        std::cout << "Creating CSV reader..." << std::endl;
        csv::CSVReader reader("/home/moinshaikh/CLionProjects/LibtorchOpenCVTutorials/databases/fashion-mnist_train.csv", format);
        std::cout << "CSV reader created successfully" << std::endl;
        
        int count = 0;
        for (auto it = reader.begin(); it != reader.end() && count < 3; ++it) {
            const auto& row = *it;
            std::cout << "Row " << count << " has " << row.size() << " columns" << std::endl;
            for (int i = 0; i < std::min(5, (int)row.size()); ++i) {
                std::cout << "  Col " << i << ": '" << row[i].get<std::string>() << "'" << std::endl;
            }
            count++;
        }
        std::cout << "CSV reading completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        throw;
    }
}
*/