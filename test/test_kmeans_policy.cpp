#include "../include/kmeans_policy.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>

namespace {

class KMeansPolicyFixture : public ::testing::TestWithParam<bool> {
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
    gmix::KMeansParameters<2> parameters;
    parameters.n_components = 2;
    parameters.n_iterations = 10;
    parameters.early_stopping_threshold = 0.0;
    parameters.warm_start = GetParam();
    parameters_ = std::move(parameters);
  }

  static gmix::GaussianMixture<2> gmm_;
  static gmix::StaticRowsMatrix<2> samples_;
  gmix::KMeansParameters<2> parameters_{};
};

gmix::GaussianMixture<2> KMeansPolicyFixture::gmm_{};
gmix::StaticRowsMatrix<2> KMeansPolicyFixture::samples_{};

TEST_P(
    KMeansPolicyFixture,
    Fit_GivenParametersAndSamples_ExpectCorrectApproximationOfUnderlyingDistribution) {
  gmix::GaussianMixture<2, gmix::KMeansPolicy> gmm{parameters_};
  if (GetParam()) {
    gmm.add_component(
        {0.5, (gmix::ColVector<2>() << 1.0, 1.0).finished(),
         (gmix::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
    gmm.add_component(
        {0.5, (gmix::ColVector<2>() << -1.0, -1.0).finished(),
         (gmix::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
  }

  gmm.fit(samples_);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(gmm_, gmm, test::RANDOM_TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(KMeansPolicyWarmColdStart, KMeansPolicyFixture,
                         testing::Values(true, false));

} // namespace
