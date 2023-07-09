#include "../include/ExpectationMaximizationStrategy.hpp"
#include "TestHelpers.hpp"
#include <gtest/gtest.h>

namespace {

TEST(EvaluateResponsibilities, Dummy) {}

TEST(EstimateParameters, Dummy) {}

class ExpectationMaximizationFixture : public ::testing::TestWithParam<bool> {
protected:
  static void SetUpTestSuite() {
    gmm_.add_component({0.5, (gm::Vector<2>() << 2.0, 9.0).finished(),
                        (gm::Matrix<2, 2>() << 2.0, 0.0, 0.0, 2.0).finished()});
    gmm_.add_component({0.5, (gm::Vector<2>() << -5.0, 4.0).finished(),
                        (gm::Matrix<2, 2>() << 0.5, 0.0, 0.0, 0.5).finished()});
    samples_ = gm::draw_from_gaussian_mixture(gmm_, 1E5);
  }

  void SetUp() override {
    gm::ExpectationMaximizationParameters<2> parameters;
    parameters.n_components = 2;
    parameters.n_iterations = 10;
    parameters.early_stopping_threshold = 0.0;
    parameters.warm_start = GetParam();
    strategy_ = gm::ExpectationMaximizationStrategy<2>{parameters};
  }

  static gm::GaussianMixture<2> gmm_;
  static gm::StaticRowsMatrix<2> samples_;
  gm::ExpectationMaximizationParameters<2> parameters_{};
  gm::ExpectationMaximizationStrategy<2> strategy_{
      gm::ExpectationMaximizationParameters<2>{}};
};

gm::GaussianMixture<2> ExpectationMaximizationFixture::gmm_{};
gm::StaticRowsMatrix<2> ExpectationMaximizationFixture::samples_{};

TEST_P(
    ExpectationMaximizationFixture,
    Fit_GivenParametersAndSamples_ExpectCorrectApproximationOfUnderlyingDistribution) {
  gm::GaussianMixture<2> gmm;
  if (GetParam()) {
    gmm.add_component({0.5, (gm::Vector<2>() << 1.0, 1.0).finished(),
                       (gm::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
    gmm.add_component({0.5, (gm::Vector<2>() << -1.0, -1.0).finished(),
                       (gm::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
  }

  gm::fit(samples_, strategy_, gmm);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(gmm_, gmm, test::RANDOM_TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(ExpectationMaximizationStrategyWarmColdStart,
                         ExpectationMaximizationFixture,
                         testing::Values(true, false));

} // namespace
