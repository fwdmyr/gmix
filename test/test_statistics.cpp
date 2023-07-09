#include "../include/Statistics.hpp"
#include "TestHelpers.hpp"
#include <gtest/gtest.h>

namespace {

class StatisticsFixture : public testing::Test {
protected:
  static void SetUpTestSuite() {
    gm::GaussianMixture<2> gmm;
    mean_ = (gm::ColVector<2>() << 9.7, -4.2).finished();
    covariance_ = (gm::Matrix<2, 2>() << 3.2, 0.2, 0.7, 2.5).finished();
    gmm.add_component({1.0, mean_, covariance_});
    samples_ = gm::draw_from_gaussian_mixture(gmm, 100000);
  }

  static gm::ColVector<2> mean_;
  static gm::Matrix<2, 2> covariance_;
  static gm::StaticRowsMatrix<2> samples_;
};

gm::ColVector<2> StatisticsFixture::mean_{};
gm::Matrix<2, 2> StatisticsFixture::covariance_{};
gm::StaticRowsMatrix<2> StatisticsFixture::samples_{};

TEST_F(
    StatisticsFixture,
    SampleMean_GivenSetOfSamplesFromDistribution_ExpectCorrectApproximationOfDistributionMean) {
  const auto mean = gm::internal::sample_mean(samples_);

  EXPECT_TRUE(test::is_near(mean, mean_, test::RANDOM_TOLERANCE));
}

TEST_F(
    StatisticsFixture,
    SampleCovariance_GivenSetOfSamplesFromDistribution_ExpectCorrectApproximationOfDistributionCovariance) {
  const auto covariance = gm::internal::sample_covariance(samples_);

  std::cerr << covariance << std::endl;

  EXPECT_TRUE(test::is_near(covariance, covariance_, test::RANDOM_TOLERANCE));
}

TEST_F(
    StatisticsFixture,
    SampleCovariance_GivenSetOfSamplesFromDistributionAndDistributionMean_ExpectCorrectApproximationOfDistributionCovariance) {
  const auto covariance = gm::internal::sample_covariance(samples_, mean_);

  EXPECT_TRUE(test::is_near(covariance, covariance_, test::RANDOM_TOLERANCE));
}

} // namespace
