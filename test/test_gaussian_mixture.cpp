#include "../include/KMeansStrategy.hpp"
#include "../include/Statistics.hpp"
#include "TestHelpers.hpp"
#include <gtest/gtest.h>
#include <numeric>

// TODO: Increase test coverage, use parametric tests, use test fixtures
// TODO: Since helper functions now live in gm::internal, write unit tests for
// them too

TEST(
    DrawFromGaussianMixture,
    GivenSamplesDrawnFromUnivariateGaussianMixture_ExpectCorrectReconstructionOfGaussianMixture) {

  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixture<1> gmm;
  const auto weight = 1.0;
  const auto mean = (gm::Vector<1>() << 9.7).finished();
  const auto covariance = (gm::Matrix<1, 1>() << 1.7).finished();
  gmm.add_component({weight, mean, covariance});

  const auto samples = gm::draw_from_gaussian_mixture(gmm, N_SAMPLES);

  const auto sample_mean = gm::internal::sample_mean(samples);
  const auto sample_covariance =
      gm::internal::sample_covariance(samples, sample_mean);

  EXPECT_TRUE(test::is_near(mean, sample_mean, test::RANDOM_TOLERANCE));
  EXPECT_TRUE(
      test::is_near(covariance, sample_covariance, test::RANDOM_TOLERANCE));
}

TEST(
    DrawFromGaussianMixture,
    GivenSamplesDrawnFromMultivariateGaussianMixture_ExpectCorrectReconstructionOfGaussianMixture) {

  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixture<2> gmm;
  const auto weight = 1.0;
  const auto mean = (gm::Vector<2>() << 9.7, 4.2).finished();
  const auto covariance = (gm::Matrix<2, 2>() << 1.7, 0.0, 0.0, 2.4).finished();
  gmm.add_component({weight, mean, covariance});

  const auto samples = gm::draw_from_gaussian_mixture(gmm, N_SAMPLES);

  const auto sample_mean = gm::internal::sample_mean(samples);
  const auto sample_covariance =
      gm::internal::sample_covariance(samples, sample_mean);

  EXPECT_TRUE(test::is_near(mean, sample_mean, test::RANDOM_TOLERANCE));
  EXPECT_TRUE(
      test::is_near(covariance, sample_covariance, test::RANDOM_TOLERANCE));
}
