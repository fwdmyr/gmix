#include "../include/expectation_maximization_strategy.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>

namespace {

TEST(EvaluateResponsibilities,
     GivenSamplesAndGaussianMixture_ExpectCorrectResponsibilityMatrix) {
  const auto samples =
      (gmix::StaticRowsMatrix<2>(2, 2) << -3.0, -2.0, 4.0, 3.0).finished();
  auto gmm = gmix::GaussianMixture<2>{};
  gmm.add_component({0.5, (gmix::ColVector<2>() << 2.0, 9.0).finished(),
                     (gmix::Matrix<2, 2>() << 2.0, 0.0, 0.0, 2.0).finished()});
  gmm.add_component({0.5, (gmix::ColVector<2>() << -5.0, 4.0).finished(),
                     (gmix::Matrix<2, 2>() << 0.5, 0.0, 0.0, 0.5).finished()});
  auto responsibilities =
      (gmix::MatrixX(2, 2) << 0.0, 0.0, 0.0, 0.0).finished();

  std::ignore = gmix::internal::evaluate_responsibilities(
      gmm.get_components(), samples, responsibilities);

  EXPECT_NEAR(responsibilities(0, 0), 0.000050864504923,
              test::DETERMINISTIC_TOLERANCE);
  EXPECT_NEAR(responsibilities(0, 1), 0.012293749653344,
              test::DETERMINISTIC_TOLERANCE);
  EXPECT_NEAR(responsibilities.col(0).sum(), 1.0,
              test::DETERMINISTIC_TOLERANCE);
  EXPECT_NEAR(responsibilities.col(1).sum(), 1.0,
              test::DETERMINISTIC_TOLERANCE);
}

TEST(
    EstimateParameters,
    GivenSamplesResponsibilitiesAndGaussianMixture_ExpectCorrectUpdateOfParameters) {
  const auto samples =
      (gmix::StaticRowsMatrix<2>(2, 2) << 1.0, 2.0, 3.0, 4.0).finished();
  const auto responsibilities =
      (gmix::MatrixX(2, 2) << 0.3, 0.6, 0.7, 0.4).finished();
  auto gmm = gmix::GaussianMixture<2>{};
  gmm.add_component({0.5, (gmix::ColVector<2>() << 2.0, 9.0).finished(),
                     (gmix::Matrix<2, 2>() << 2.0, 0.0, 0.0, 2.0).finished()});
  gmm.add_component({0.5, (gmix::ColVector<2>() << -5.0, 4.0).finished(),
                     (gmix::Matrix<2, 2>() << 0.5, 0.0, 0.0, 0.5).finished()});

  auto gmm_expected = gmix::GaussianMixture<2>{};
  gmm_expected.add_component(
      {0.45,
       (gmix::ColVector<2>() << 1.6666666666666667, 3.6666666666666667)
           .finished(),
       (gmix::Matrix<2, 2>() << 0.2222222222222222, 0.2222222222222222,
        0.2222222222222222, 0.2222222222222222)
           .finished()});
  gmm_expected.add_component(
      {0.55,
       (gmix::ColVector<2>() << 1.3636363636363636, 3.3636363636363636)
           .finished(),
       (gmix::Matrix<2, 2>() << 0.2314049586776859, 0.2314049586776859,
        0.2314049586776859, 0.2314049586776859)
           .finished()});

  gmix::internal::estimate_parameters(samples, responsibilities,
                                      gmm.get_components());

  EXPECT_TRUE(test::compare_gaussian_mixtures(gmm, gmm_expected,
                                              test::DETERMINISTIC_TOLERANCE));
}

class ExpectationMaximizationFixture : public ::testing::TestWithParam<bool> {
protected:
  static void SetUpTestSuite() {
    gmm_.add_component(
        {0.5, (gmix::ColVector<2>() << 2.0, 9.0).finished(),
         (gmix::Matrix<2, 2>() << 2.0, 0.0, 0.0, 2.0).finished()});
    gmm_.add_component(
        {0.5, (gmix::ColVector<2>() << -5.0, 4.0).finished(),
         (gmix::Matrix<2, 2>() << 0.5, 0.0, 0.0, 0.5).finished()});
    samples_ = gmix::draw_from_gaussian_mixture(gmm_, 1E5);
  }

  void SetUp() override {
    gmix::ExpectationMaximizationParameters<2> parameters;
    parameters.n_components = 2;
    parameters.n_iterations = 10;
    parameters.early_stopping_threshold = 0.0;
    parameters.warm_start = GetParam();
    strategy_ = gmix::ExpectationMaximizationStrategy<2>{parameters};
  }

  static gmix::GaussianMixture<2> gmm_;
  static gmix::StaticRowsMatrix<2> samples_;
  gmix::ExpectationMaximizationParameters<2> parameters_{};
  gmix::ExpectationMaximizationStrategy<2> strategy_{
      gmix::ExpectationMaximizationParameters<2>{}};
};

gmix::GaussianMixture<2> ExpectationMaximizationFixture::gmm_{};
gmix::StaticRowsMatrix<2> ExpectationMaximizationFixture::samples_{};

TEST_P(
    ExpectationMaximizationFixture,
    Fit_GivenParametersAndSamples_ExpectCorrectApproximationOfUnderlyingDistribution) {
  gmix::GaussianMixture<2> gmm;
  if (GetParam()) {
    gmm.add_component(
        {0.5, (gmix::ColVector<2>() << 1.0, 1.0).finished(),
         (gmix::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
    gmm.add_component(
        {0.5, (gmix::ColVector<2>() << -1.0, -1.0).finished(),
         (gmix::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
  }

  gmix::fit(samples_, strategy_, gmm);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(gmm_, gmm, test::RANDOM_TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(ExpectationMaximizationStrategyWarmColdStart,
                         ExpectationMaximizationFixture,
                         testing::Values(true, false));

} // namespace
