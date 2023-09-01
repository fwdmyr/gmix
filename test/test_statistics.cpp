#include "../include/statistics.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>

namespace {

class StatisticsFixture : public testing::Test {
protected:
  static void SetUpTestSuite() {
    gmix::GaussianMixture<2> gmm;
    mean_ = (gmix::ColVector<2>() << 9.7, -4.2).finished();
    covariance_ = (gmix::Matrix<2, 2>() << 1.2, 0.0, 0.0, 1.2).finished();
    gmm.add_component({1.0, mean_, covariance_});
    samples_ = gmix::draw_from_gaussian_mixture(gmm, 100000);
  }

  static gmix::ColVector<2> mean_;
  static gmix::Matrix<2, 2> covariance_;
  static gmix::StaticRowsMatrix<2> samples_;
};

gmix::ColVector<2> StatisticsFixture::mean_{};
gmix::Matrix<2, 2> StatisticsFixture::covariance_{};
gmix::StaticRowsMatrix<2> StatisticsFixture::samples_{};

TEST_F(
    StatisticsFixture,
    SampleMean_GivenSetOfSamplesFromDistribution_ExpectCorrectApproximationOfDistributionMean) {
  const auto mean = gmix::internal::sample_mean(samples_);

  EXPECT_TRUE(test::is_near(mean, mean_, test::RANDOM_TOLERANCE));
}

TEST_F(
    StatisticsFixture,
    SampleCovariance_GivenSetOfSamplesFromDistribution_ExpectCorrectApproximationOfDistributionCovariance) {
  const auto covariance = gmix::internal::sample_covariance(samples_);

  std::cerr << covariance << std::endl;

  EXPECT_TRUE(test::is_near(covariance, covariance_, test::RANDOM_TOLERANCE));
}

TEST_F(
    StatisticsFixture,
    SampleCovariance_GivenSetOfSamplesFromDistributionAndDistributionMean_ExpectCorrectApproximationOfDistributionCovariance) {
  const auto covariance = gmix::internal::sample_covariance(samples_, mean_);

  EXPECT_TRUE(test::is_near(covariance, covariance_, test::RANDOM_TOLERANCE));
}

} // namespace
