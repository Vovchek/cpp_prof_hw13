#include "helpers.h"

#include <fstream>
#include <sstream>
#include <iterator>

namespace mnist {

Eigen::MatrixXf read_mat_from_stream(size_t rows, size_t cols, std::istream& stream) {
    Eigen::MatrixXf res(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float val;
            stream >> val;
            res(i, j) = val;
        }
    }
    return res;
}

Eigen::MatrixXf read_mat_from_file(size_t rows, size_t cols, const std::string& filepath) {
    std::ifstream stream{filepath};
    if(!stream.is_open()) {
        std::stringstream ss;
        ss << "read_mat_from_file: Error opening " << filepath;
        throw std::runtime_error{ss.str()};
    }
    return read_mat_from_stream(rows, cols, stream);
}

bool read_features(std::istream& stream, Classifier::features_t& features, const char sep) {
    std::string line;
    std::getline(stream, line);
    if(sep != ' ')
        std::replace(line.begin(), line.end(), sep, ' ');
    features.clear();
    std::istringstream linestream{line};
    double value;
    while (linestream >> value) {
        features.push_back(value);
    }
    return stream.good();
}

std::vector<float> read_vector(std::istream& stream) {
    std::vector<float> result;

    std::copy(std::istream_iterator<float>(stream),
              std::istream_iterator<float>(),
              std::back_inserter(result));
    return result;
}

}