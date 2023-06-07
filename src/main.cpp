#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>

#include <Eigen/Dense>

#include <mlp_classifier.h>
#include <helpers.h>

using namespace mnist;

const size_t input_dim = 784;
const size_t hidden_dim = 128;
const size_t output_dim = 10;

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    fprintf(stderr, "Usage: %s <test_data_file>\n", argv[0]);
    return 1;
  }

  try
  {
    auto w1 = read_mat_from_file(input_dim, hidden_dim, "train/w1.txt");
    auto w2 = read_mat_from_file(hidden_dim, output_dim, "train/w2.txt");

    auto clf = MlpClassifier{w1.transpose(), w2.transpose()};

    auto features = MlpClassifier::features_t{};
    std::ifstream test_data{argv[1]};

    if (!test_data.is_open())
    {
      std::stringstream ss;
      ss << "read_mat_from_file: Error opening " << argv[1];
      throw std::runtime_error{ss.str()};
    }

    size_t hits{0};
    size_t samples{0};

    std::vector<size_t> TP(output_dim);
    std::vector<size_t> FP(output_dim);
    std::vector<size_t> FN(output_dim);

    for (; !test_data.eof(); ++samples)
    {
      size_t y;
      test_data >> y;
      if (!read_features(test_data, features, ','))
      {
        break;
      }
      auto yp = clf.predict(features);
      if (y == yp) {
        ++TP[y];
        ++hits;
      } else {
        ++FP[yp];
        ++FN[y];
      }
    }

    auto rec ([](size_t tp, size_t fn) {return (tp+fn) ? float(tp)/(tp+fn) : 0.0f;});
    auto prec ([](size_t tp, size_t fp) {return (tp+fp) ? float(tp)/(tp+fp) : 0.0f;});

    std::cout << std::setprecision(3) << std::setw(6) << "class"
              << std::setw(10) << "precision"
              << std::setw(10)<< "recall"
              << '\n';
    for(auto item{0}; item < output_dim; ++item) {
      std::cout << std::setw(6) << item
                << std::setw(10) << prec(TP[item], FN[item])
                << std::setw(10) << rec(TP[item], FP[item])
                << '\n';
    }

    std::cout << "accuracy : " << (samples ? (float(hits) / samples) : 0.0) << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
  }
}