#include "../include/Statistics.hpp"
#include <gtest/gtest.h>
#include <numeric>

#define GTEST_COUT std::cerr << "[          ] [ INFO ]"

// Tests if mean of sampling distribution approximates mean of underlying
// distribution.
TEST(SamplingTest, SingleUnivariateComponentMeanTest) {

  constexpr int DIM = 1;
  constexpr size_t N_SAMPLES = 1E6;
  gm::GaussianMixture<DIM> gmm;
  const auto weight = 1.0;
  const auto mean = (gm::Vector<DIM>() << 9.7).finished();
  const auto covariance = (gm::Matrix<DIM, DIM>() << 1.7).finished();
  gmm.add_component({weight, mean, covariance});

  const auto samples = gm::draw_from_gaussian_mixture(gmm, N_SAMPLES);

  const auto sample_mean = gm::sample_mean(samples);

  EXPECT_NEAR(mean(0), sample_mean(0), 1E-2);
}

// Tests if covariance of sampling distribution approximates covariance of
// underlying distribution.
TEST(SamplingTest, SingleUnivariateComponentCovarianceTest) {

  constexpr int DIM = 1;
  constexpr size_t N_SAMPLES = 1E6;
  gm::GaussianMixture<DIM> gmm;
  const auto weight = 1.0;
  const auto mean = (gm::Vector<DIM>() << 9.7).finished();
  const auto covariance = (gm::Matrix<DIM, DIM>() << 1.7).finished();
  gmm.add_component({weight, mean, covariance});

  const auto samples = gm::draw_from_gaussian_mixture(gmm, N_SAMPLES);

  const auto sample_mean = gm::sample_mean(samples);

  const auto sample_covariance = gm::sample_covariance(samples, sample_mean);

  EXPECT_NEAR(covariance(0, 0), sample_covariance(0, 0), 1E-2);
}

// Tests if mean of sampling distribution approximates mean of underlying
// distribution.
TEST(SamplingTest, SingleMultivariateComponentMeanTest) {

  constexpr int DIM = 2;
  constexpr size_t N_SAMPLES = 1E6;
  gm::GaussianMixture<DIM> gmm;
  const auto weight = 1.0;
  const auto mean = (gm::Vector<DIM>() << 9.7, 4.2).finished();
  const auto covariance =
      (gm::Matrix<DIM, DIM>() << 1.7, 0.0, 0.0, 2.4).finished();
  gmm.add_component({weight, mean, covariance});

  const auto samples = gm::draw_from_gaussian_mixture(gmm, N_SAMPLES);

  const auto sample_mean = gm::sample_mean(samples);

  EXPECT_NEAR(mean(0), sample_mean(0), 1E-2);
  EXPECT_NEAR(mean(1), sample_mean(1), 1E-2);
}

// Tests if covariance of sampling distribution approximates covariance of
// underlying distribution.
TEST(SamplingTest, SingleMultivariateComponentCovarianceTest) {

  constexpr int DIM = 2;
  constexpr size_t N_SAMPLES = 1E6;
  gm::GaussianMixture<DIM> gmm;
  const auto weight = 1.0;
  const auto mean = (gm::Vector<DIM>() << 9.7, 4.2).finished();
  const auto covariance =
      (gm::Matrix<DIM, DIM>() << 1.7, 0.0, 0.0, 2.4).finished();
  gmm.add_component({weight, mean, covariance});

  const auto samples = gm::draw_from_gaussian_mixture(gmm, N_SAMPLES);

  const auto sample_mean = gm::sample_mean(samples);

  const auto sample_covariance = gm::sample_covariance(samples, sample_mean);

  EXPECT_NEAR(covariance(0, 0), sample_covariance(0, 0), 1E-2);
  EXPECT_NEAR(covariance(1, 1), sample_covariance(1, 1), 1E-2);
}
