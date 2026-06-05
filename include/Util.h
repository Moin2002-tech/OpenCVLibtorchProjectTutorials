//
// Created by moinshaikh on 5/25/26.
//

#ifndef LIBTORCHOPENCVTUTORIALS_UTIL_H
#define LIBTORCHOPENCVTUTORIALS_UTIL_H

#include <iostream>
#include <torch/torch.h>
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
#endif //LIBTORCHOPENCVTUTORIALS_UTIL_H
