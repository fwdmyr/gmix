#include "../include/GaussianMixture.hpp"
#include "../include/KMeansStrategy.hpp"
#include "../include/Statistics.hpp"
#include <gtest/gtest.h>
#include <numeric>

#define GTEST_COUT std::cerr << "[          ] [ INFO ]"

// Tests if mean of sampling distribution approximates mean of underlying
// distribution.
TEST(FittingTest, kMeansStrategyFittingTest) {

  constexpr int DIM = 2;
  constexpr size_t N_SAMPLES = 1E6;
  gm::GaussianMixture<DIM> gmm;
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 2.0, 9.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -5.0, 2.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(gmm, N_SAMPLES);

  gm::KMeansStrategyParameters parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  gm::KMeansStrategy<DIM> k_means{parameters};
  gmm.set_strategy(std::move(k_means));
  gmm.fit(samples);

  EXPECT_NEAR(1, 1, 1E-2);
}