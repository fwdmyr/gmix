#include "TestHelpers.hpp"
#include <gtest/gtest.h>

namespace {

class StatisticsTestFixture : public testing::Test {
protected:
  void SetUp() override {
    gm::GaussianMixture<2U> gmm;
    mean_ = (gm::Vector<2U>() << 9.7, -4.2).finished();
    covariance_ = (gm::Matrix<2U, 2U>() << 5.4, 1.3, 2.1, 8.7).finished();
    gmm.add_component({1.0, mean_, covariance_});

    samples_ = gm::draw_from_gaussian_mixture(gmm, 100000U);
  }

  gm::Vector<2U> mean_;
  gm::Matrix<2U, 2U> covariance_;
  gm::StaticRowsMatrix<2U> samples_;
};

} // namespace

TEST_F(StatisticsTestFixture, TestSampleMeanCalculation) {
  const auto mean = gm::internal::sample_mean(samples_);

  EXPECT_TRUE(test::is_near(mean, mean_, test::RANDOM_TOLERANCE));
}

TEST_F(StatisticsTestFixture, TestSampleCovarianceCalculation) {
  const auto covariance = gm::internal::sample_covariance(samples_);

  EXPECT_TRUE(test::is_near(covariance, covariance_, test::RANDOM_TOLERANCE));
}

TEST_F(StatisticsTestFixture, TestSampleCovarianceCalculationGivenMean) {
  const auto covariance = gm::internal::sample_covariance(samples_, mean_);

  EXPECT_TRUE(test::is_near(covariance, covariance_, test::RANDOM_TOLERANCE));
}
