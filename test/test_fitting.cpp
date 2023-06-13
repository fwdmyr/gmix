#include "../include/GaussianMixture.hpp"
#include "../include/KMeansStrategy.hpp"
#include "../include/Statistics.hpp"
#include <gtest/gtest.h>
#include <numeric>

#define GTEST_COUT std::cerr << "[          ] [ INFO ]"

// Tests if kMeans algorithm approximates the underlying Gaussian mixture
// distribution's weight correctly.
TEST(FittingTest, kMeansStrategyWeightFittingTest) {

  constexpr int DIM = 2;
  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixtureKMeans<DIM> gmm;
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 2.0, 9.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -5.0, 4.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(gmm, N_SAMPLES);

  gmm.reset();
  gm::KMeansStrategy<DIM>::Parameters parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  gmm.set_strategy(parameters);
  gmm.fit(samples);

  EXPECT_NEAR(gmm.get_component(0).get_weight(), 0.5, 5E-2);
  EXPECT_NEAR(gmm.get_component(1).get_weight(), 0.5, 5E-2);
}

// Tests if kMeans algorithm approximates the underlying Gaussian mixture
// distribution's mean correctly.
TEST(FittingTest, kMeansStrategyMeanFittingTest) {

  constexpr int DIM = 2;
  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixtureKMeans<DIM> gmm;
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 2.0, 9.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -5.0, 4.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(gmm, N_SAMPLES);

  gmm.reset();
  gm::KMeansStrategy<DIM>::Parameters parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  gmm.set_strategy(parameters);
  gmm.fit(samples);

  EXPECT_NEAR(gmm.get_component(0).get_mean()(0), 2.0, 5E-2);
  EXPECT_NEAR(gmm.get_component(0).get_mean()(1), 9.0, 5E-2);
  EXPECT_NEAR(gmm.get_component(1).get_mean()(0), -5.0, 5E-2);
  EXPECT_NEAR(gmm.get_component(1).get_mean()(1), 4.0, 5E-2);
}

// Tests if kMeans algorithm approximates the underlying Gaussian mixture
// distribution's covariance correctly.
TEST(FittingTest, kMeansStrategyCovarianceFittingTest) {

  constexpr int DIM = 2;
  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixtureKMeans<DIM> gmm;
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 2.0, 9.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -5.0, 4.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(gmm, N_SAMPLES);

  gmm.reset();
  gm::KMeansStrategy<DIM>::Parameters parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  gmm.set_strategy(parameters);
  gmm.fit(samples);

  EXPECT_NEAR(gmm.get_component(0).get_covariance()(0, 0), 1.0, 5E-2);
  EXPECT_NEAR(gmm.get_component(0).get_covariance()(1, 0), 0.0, 5E-2);
  EXPECT_NEAR(gmm.get_component(0).get_covariance()(0, 1), 0.0, 5E-2);
  EXPECT_NEAR(gmm.get_component(0).get_covariance()(1, 1), 1.0, 5E-2);
  EXPECT_NEAR(gmm.get_component(1).get_covariance()(0, 0), 1.0, 5E-2);
  EXPECT_NEAR(gmm.get_component(1).get_covariance()(1, 0), 0.0, 5E-2);
  EXPECT_NEAR(gmm.get_component(1).get_covariance()(0, 1), 0.0, 5E-2);
  EXPECT_NEAR(gmm.get_component(1).get_covariance()(1, 1), 1.0, 5E-2);
}
