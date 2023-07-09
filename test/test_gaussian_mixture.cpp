#include "../include/KMeansStrategy.hpp"
#include "../include/Statistics.hpp"
#include "TestHelpers.hpp"
#include <gtest/gtest.h>
#include <numeric>

TEST(
    DrawFromGaussianMixture,
    GivenSamplesDrawnFromUnivariateGaussianMixture_ExpectCorrectReconstructionOfGaussianMixture) {
  gm::GaussianMixture<1> gmm;
  const auto weight = 1.0;
  const auto mean = gm::initialize<gm::ColVector<1>>({{9.7}});
  const auto covariance = gm::initialize<gm::Matrix<1, 1>>({{1.7}});
  gmm.add_component({weight, mean, covariance});

  const auto samples = gm::draw_from_gaussian_mixture(gmm, 1E5);

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
  gm::GaussianMixture<2> gmm;
  const auto weight = 1.0;
  const auto mean = gm::initialize<gm::ColVector<2>>({9.7, 4.2});
  const auto covariance =
      gm::initialize<gm::Matrix<2, 2>>({{1.7, 0.0}, {0.0, 2.4}});
  gmm.add_component({weight, mean, covariance});

  const auto samples = gm::draw_from_gaussian_mixture(gmm, 1E5);

  const auto sample_mean = gm::internal::sample_mean(samples);
  const auto sample_covariance =
      gm::internal::sample_covariance(samples, sample_mean);

  EXPECT_TRUE(test::is_near(mean, sample_mean, test::RANDOM_TOLERANCE));
  EXPECT_TRUE(
      test::is_near(covariance, sample_covariance, test::RANDOM_TOLERANCE));
}
