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

  constexpr int DIM = 2;
  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixtureKMeans<DIM> sample_gmm;
  sample_gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 2.0, 9.0).finished(),
       (gm::Matrix<DIM, DIM>() << 2.0, 0.0, 0.0, 2.0).finished()});
  sample_gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -5.0, 4.0).finished(),
       (gm::Matrix<DIM, DIM>() << 0.5, 0.0, 0.0, 0.5).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(sample_gmm, N_SAMPLES);

  gm::GaussianMixtureKMeans<DIM> gmm;
  gm::KMeansStrategy<DIM>::Parameters parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  parameters.warm_start = false;
  gmm.set_strategy(parameters);
  gmm.fit(samples);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(sample_gmm, gmm, test::TOLERANCE));
}

TEST(FittingTest, kMeansStrategyFittingWarmStartTest) {

  constexpr int DIM = 2;
  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixtureKMeans<DIM> sample_gmm;
  sample_gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 2.0, 9.0).finished(),
       (gm::Matrix<DIM, DIM>() << 2.0, 0.0, 0.0, 2.0).finished()});
  sample_gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -5.0, 4.0).finished(),
       (gm::Matrix<DIM, DIM>() << 0.5, 0.0, 0.0, 0.5).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(sample_gmm, N_SAMPLES);

  gm::GaussianMixtureKMeans<DIM> gmm;
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 1.0, 1.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -1.0, -1.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gm::KMeansStrategy<DIM>::Parameters parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  parameters.warm_start = true;
  gmm.set_strategy(parameters);
  gmm.fit(samples);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(sample_gmm, gmm, test::TOLERANCE));
}

TEST(FittingTest, ExpectationMaximizationStrategyFittingColdStartTest) {

  constexpr int DIM = 2;
  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixtureExpectationMaximization<DIM> sample_gmm;
  sample_gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 2.0, 9.0).finished(),
       (gm::Matrix<DIM, DIM>() << 2.0, 0.0, 0.0, 2.0).finished()});
  sample_gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -5.0, 4.0).finished(),
       (gm::Matrix<DIM, DIM>() << 0.5, 0.0, 0.0, 0.5).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(sample_gmm, N_SAMPLES);

  gm::GaussianMixtureExpectationMaximization<DIM> gmm;
  gm::ExpectationMaximizationStrategy<DIM>::Parameters parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  parameters.warm_start = false;
  gmm.set_strategy(parameters);
  gmm.fit(samples);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(sample_gmm, gmm, test::TOLERANCE));
}

TEST(FittingTest, ExpectationMaximizationStrategyFittingWarmStartTest) {

  constexpr int DIM = 2;
  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixtureExpectationMaximization<DIM> sample_gmm;
  sample_gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 2.0, 9.0).finished(),
       (gm::Matrix<DIM, DIM>() << 2.0, 0.0, 0.0, 2.0).finished()});
  sample_gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -5.0, 4.0).finished(),
       (gm::Matrix<DIM, DIM>() << 0.5, 0.0, 0.0, 0.5).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(sample_gmm, N_SAMPLES);

  gm::GaussianMixtureExpectationMaximization<DIM> gmm;
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 1.0, 1.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -1.0, -1.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gm::ExpectationMaximizationStrategy<DIM>::Parameters parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  parameters.warm_start = true;
  gmm.set_strategy(parameters);
  gmm.fit(samples);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(sample_gmm, gmm, test::TOLERANCE));
}

TEST(FittingTest, VariationalBayesianInferenceStrategyFittingColdStartTest) {

  constexpr int DIM = 2;
  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixtureVariationalBayesianInference<DIM> sample_gmm;
  sample_gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 2.0, 9.0).finished(),
       (gm::Matrix<DIM, DIM>() << 2.0, 0.0, 0.0, 2.0).finished()});
  sample_gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -5.0, 4.0).finished(),
       (gm::Matrix<DIM, DIM>() << 0.5, 0.0, 0.0, 0.5).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(sample_gmm, N_SAMPLES);

  gm::GaussianMixtureVariationalBayesianInference<DIM> gmm;
  gm::VariationalBayesianInferenceStrategy<DIM>::Parameters parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  parameters.warm_start = false;
  parameters.dirichlet_prior_weight = 1.0;
  parameters.normal_prior_mean = (gm::Vector<DIM>() << 0.0, 0.0).finished();
  parameters.normal_prior_covariance_scaling = 1.0;
  parameters.wishart_prior_degrees_of_freedom = 2.0;
  parameters.wishart_prior_information =
      (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished();

  gmm.set_strategy(parameters);
  gmm.fit(samples);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(sample_gmm, gmm, test::TOLERANCE));
}

TEST(FittingTest, VariationalBayesianInferenceStrategyFittingWarmStartTest) {

  constexpr int DIM = 2;
  constexpr size_t N_SAMPLES = 1E5;
  gm::GaussianMixtureVariationalBayesianInference<DIM> sample_gmm;
  sample_gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 2.0, 9.0).finished(),
       (gm::Matrix<DIM, DIM>() << 2.0, 0.0, 0.0, 2.0).finished()});
  sample_gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -5.0, 4.0).finished(),
       (gm::Matrix<DIM, DIM>() << 0.5, 0.0, 0.0, 0.5).finished()});
  const auto samples = gm::draw_from_gaussian_mixture(sample_gmm, N_SAMPLES);

  gm::GaussianMixtureVariationalBayesianInference<DIM> gmm;
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << 1.0, 1.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gmm.add_component(
      {0.5, (gm::Vector<DIM>() << -1.0, -1.0).finished(),
       (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gm::VariationalBayesianInferenceStrategy<DIM>::Parameters parameters;
  parameters.n_components = 2;
  parameters.n_iterations = 10;
  parameters.warm_start = false;
  parameters.dirichlet_prior_weight = 1.0;
  parameters.normal_prior_mean = (gm::Vector<DIM>() << 0.0, 0.0).finished();
  parameters.normal_prior_covariance_scaling = 1.0;
  parameters.wishart_prior_degrees_of_freedom = 2.0;
  parameters.wishart_prior_information =
      (gm::Matrix<DIM, DIM>() << 1.0, 0.0, 0.0, 1.0).finished();

  gmm.set_strategy(parameters);
  gmm.fit(samples);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(sample_gmm, gmm, test::TOLERANCE));
}
