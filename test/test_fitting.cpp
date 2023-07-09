#include "../include/ExpectationMaximizationStrategy.hpp"
#include "../include/KMeansStrategy.hpp"
#include "../include/Statistics.hpp"
#include "../include/VariationalBayesianInferenceStrategy.hpp"
#include "TestHelpers.hpp"
#include <numeric>

// TODO: Increase test coverage, use parametric tests, use test fixtures
// TODO: Since helper functions now live in gm::internal, write unit tests for
// them too

TEST(FittingTest, kMeansStrategyFittingColdStartTest) {

  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixture<2> sample_gmm;
  sample_gmm.add_component(
      {0.5, (gm::Vector<2>() << 2.0, 9.0).finished(),
       (gm::Matrix<2, 2>() << 2.0, 0.0, 0.0, 2.0).finished()});
  sample_gmm.add_component(
      {0.5, (gm::Vector<2>() << -5.0, 4.0).finished(),
       (gm::Matrix<2, 2>() << 0.5, 0.0, 0.0, 0.5).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(sample_gmm, N_SAMPLES);

  gm::GaussianMixture<2> gmm;
  gm::KMeansParameters<2> parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  parameters.early_stopping_threshold = 0.0;
  parameters.warm_start = false;
  const auto strategy = gm::KMeansStrategy<2>{parameters};

  gm::fit(samples, strategy, gmm);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(sample_gmm, gmm, test::RANDOM_TOLERANCE));
}

TEST(FittingTest, kMeansStrategyFittingWarmStartTest) {

  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixture<2> sample_gmm;
  sample_gmm.add_component(
      {0.5, (gm::Vector<2>() << 2.0, 9.0).finished(),
       (gm::Matrix<2, 2>() << 2.0, 0.0, 0.0, 2.0).finished()});
  sample_gmm.add_component(
      {0.5, (gm::Vector<2>() << -5.0, 4.0).finished(),
       (gm::Matrix<2, 2>() << 0.5, 0.0, 0.0, 0.5).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(sample_gmm, N_SAMPLES);

  gm::GaussianMixture<2> gmm;
  gmm.add_component({0.5, (gm::Vector<2>() << 1.0, 1.0).finished(),
                     (gm::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gmm.add_component({0.5, (gm::Vector<2>() << -1.0, -1.0).finished(),
                     (gm::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});

  gm::KMeansParameters<2> parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  parameters.early_stopping_threshold = 0.0;
  parameters.warm_start = true;
  const auto strategy = gm::KMeansStrategy<2>{parameters};

  gm::fit(samples, strategy, gmm);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(sample_gmm, gmm, test::RANDOM_TOLERANCE));
}

TEST(FittingTest, ExpectationMaximizationStrategyFittingColdStartTest) {

  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixture<2> sample_gmm;
  sample_gmm.add_component(
      {0.5, (gm::Vector<2>() << 2.0, 9.0).finished(),
       (gm::Matrix<2, 2>() << 2.0, 0.0, 0.0, 2.0).finished()});
  sample_gmm.add_component(
      {0.5, (gm::Vector<2>() << -5.0, 4.0).finished(),
       (gm::Matrix<2, 2>() << 0.5, 0.0, 0.0, 0.5).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(sample_gmm, N_SAMPLES);

  gm::GaussianMixture<2> gmm;
  gm::ExpectationMaximizationParameters<2> parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  parameters.early_stopping_threshold = 0.0;
  parameters.warm_start = false;
  const auto strategy = gm::ExpectationMaximizationStrategy<2>{parameters};

  gm::fit(samples, strategy, gmm);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(sample_gmm, gmm, test::RANDOM_TOLERANCE));
}

TEST(FittingTest, ExpectationMaximizationStrategyFittingWarmStartTest) {

  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixture<2> sample_gmm;
  sample_gmm.add_component(
      {0.5, (gm::Vector<2>() << 2.0, 9.0).finished(),
       (gm::Matrix<2, 2>() << 2.0, 0.0, 0.0, 2.0).finished()});
  sample_gmm.add_component(
      {0.5, (gm::Vector<2>() << -5.0, 4.0).finished(),
       (gm::Matrix<2, 2>() << 0.5, 0.0, 0.0, 0.5).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(sample_gmm, N_SAMPLES);

  gm::GaussianMixture<2> gmm;
  gmm.add_component({0.5, (gm::Vector<2>() << 1.0, 1.0).finished(),
                     (gm::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gmm.add_component({0.5, (gm::Vector<2>() << -1.0, -1.0).finished(),
                     (gm::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});

  gm::ExpectationMaximizationParameters<2> parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  parameters.early_stopping_threshold = 0.0;
  parameters.warm_start = true;
  const auto strategy = gm::ExpectationMaximizationStrategy<2>{parameters};

  gm::fit(samples, strategy, gmm);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(sample_gmm, gmm, test::RANDOM_TOLERANCE));
}

TEST(FittingTest, VariationalBayesianInferenceStrategyFittingColdStartTest) {

  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixture<2> sample_gmm;
  sample_gmm.add_component(
      {0.5, (gm::Vector<2>() << 2.0, 9.0).finished(),
       (gm::Matrix<2, 2>() << 2.0, 0.0, 0.0, 2.0).finished()});
  sample_gmm.add_component(
      {0.5, (gm::Vector<2>() << -5.0, 4.0).finished(),
       (gm::Matrix<2, 2>() << 0.5, 0.0, 0.0, 0.5).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(sample_gmm, N_SAMPLES);

  gm::GaussianMixture<2> gmm;
  gm::VariationalBayesianInferenceParameters<2> parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  parameters.early_stopping_threshold = 0.0;
  parameters.warm_start = false;
  parameters.dirichlet_prior_weight = 1.0;
  parameters.normal_prior_mean = (gm::Vector<2>() << 0.0, 0.0).finished();
  parameters.normal_prior_covariance_scaling = 1.0;
  parameters.wishart_prior_degrees_of_freedom = 2.0;
  parameters.wishart_prior_information =
      (gm::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished();
  const auto strategy = gm::VariationalBayesianInferenceStrategy<2>{parameters};

  gm::fit(samples, strategy, gmm);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(sample_gmm, gmm, test::RANDOM_TOLERANCE));
}

TEST(FittingTest, VariationalBayesianInferenceStrategyFittingWarmStartTest) {

  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixture<2> sample_gmm;
  sample_gmm.add_component(
      {0.5, (gm::Vector<2>() << 2.0, 9.0).finished(),
       (gm::Matrix<2, 2>() << 2.0, 0.0, 0.0, 2.0).finished()});
  sample_gmm.add_component(
      {0.5, (gm::Vector<2>() << -5.0, 4.0).finished(),
       (gm::Matrix<2, 2>() << 0.5, 0.0, 0.0, 0.5).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(sample_gmm, N_SAMPLES);

  gm::GaussianMixture<2> gmm;
  gmm.add_component({0.5, (gm::Vector<2>() << 1.0, 1.0).finished(),
                     (gm::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gmm.add_component({0.5, (gm::Vector<2>() << -1.0, -1.0).finished(),
                     (gm::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});

  gm::VariationalBayesianInferenceParameters<2> parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  parameters.early_stopping_threshold = 0.0;
  parameters.warm_start = false;
  parameters.dirichlet_prior_weight = 1.0;
  parameters.normal_prior_mean = (gm::Vector<2>() << 0.0, 0.0).finished();
  parameters.normal_prior_covariance_scaling = 1.0;
  parameters.wishart_prior_degrees_of_freedom = 2.0;
  parameters.wishart_prior_information =
      (gm::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished();
  const auto strategy = gm::VariationalBayesianInferenceStrategy<2>{parameters};

  gm::fit(samples, strategy, gmm);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(sample_gmm, gmm, test::RANDOM_TOLERANCE));
}
