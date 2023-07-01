#ifndef GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_HPP
#define GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_HPP

#include "BaseStrategy.hpp"
#include "Common.hpp"
#include "Statistics.hpp"
#include <eigen3/Eigen/src/Core/GlobalFunctions.h>
#include <limits>
#include <random>
#include <unsupported/Eigen/SpecialFunctions>

namespace gm {

namespace {} // namespace

template <int Dim>
class VariationalBayesianInferenceStrategy final : public BaseStrategy<Dim> {
public:
  struct Parameters {
    int n_components;
    int n_iterations;
    double dirichlet_prior_weight;
    Vector<Dim> normal_prior_mean;
    double normal_prior_covariance_scaling;
    double wishart_prior_degrees_of_freedom;
    Matrix<Dim, Dim> wishart_prior_information;
  };

  explicit VariationalBayesianInferenceStrategy(const Parameters &parameters)
      : parameters_(parameters),
        dirichlet_weight_(parameters.dirichlet_prior_weight.replicate(
            1, parameters.n_components)),
        normal_mean_(
            parameters.normal_prior_mean.replicate(1, parameters.n_components)),
        normal_covariance_scaling_(
            parameters.normal_prior_covariance_scaling.replicate(
                1, parameters.n_components)),
        wishart_information_(parameters.wishart_prior_information.replicate(
            1, parameters.n_components)),
        wishart_degrees_of_freedom_(
            parameters.wishart_prior_degrees_of_freedom.replicate(
                1, parameters.n_components)) {}
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const StaticRowsMatrix<Dim> &) const override;

private:
  Parameters parameters_{};

  // Random variables
  VectorX dirichlet_weight_{};
  StaticRowsMatrix<Dim> normal_mean_{};
  VectorX normal_covariance_scaling_{};
  StaticRowsMatrix<Dim> wishart_information_{};
  VectorX wishart_degrees_of_freedom_{};

  void iterate(const StaticRowsMatrix<Dim> &);
};

template <int Dim>
void VariationalBayesianInferenceStrategy<Dim>::fit(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) const {
  // TODO: Move this body into the iterate function and loop in this function
  const auto n_samples = samples.cols();

  // Compute Responsibilities
  auto responsibilities =
      static_cast<MatrixX>(MatrixX::Zero(parameters_.n_components, n_samples));
  for (size_t i = 0; i < parameters_.n_components; ++i) {
    const auto e_logweight = Eigen::numext::digamma(dirichlet_weight_(i)) -
                             Eigen::numext::digamma(dirichlet_weight_.sum());
    auto e_logdetcovariance =
        Dim * std::log(2) +
        std::log(
            1.0 /
            wishart_information_.block(0, i * Dim, Dim, Dim).determinant());
    for (size_t d = 1; d <= Dim; ++d) {
      e_logdetcovariance += Eigen::numext::digamma(
          0.5 * (wishart_degrees_of_freedom_(i) + 1 - d));
    }
    for (size_t j = 0; j < n_samples; ++j) {
      const auto distance = samples.col(j) - normal_mean_.col(i);
      const auto e_squaredmahalanobisdistance =
          static_cast<double>(Dim) / normal_covariance_scaling_(i) +
          wishart_degrees_of_freedom_(i) * distance.transpose() *
              wishart_information_.block(0, i * Dim, Dim, Dim).inverse() *
              distance;
      const auto logresponsibility = e_logweight + 0.5 * e_logdetcovariance -
                                     0.5 * Dim * std::log(2.0 * M_PI) -
                                     0.5 * e_squaredmahalanobisdistance;
      responsibilities(i, j) = std::exp(logresponsibility);
    }
  }
  responsibilities.colwise().normalize();

  // Compute Statistics
  const auto n_samples_responsible =
      static_cast<VectorX>(responsibilities.rowwise().sum());
  auto mu = static_cast<StaticRowsMatrix<Dim>>(
      StaticRowsMatrix<Dim>::Zero(Dim, parameters_.n_components));
  auto sigma = static_cast<StaticRowsMatrix<Dim>>(
      StaticRowsMatrix<Dim>::Zero(Dim, Dim * parameters_.n_components));
  for (size_t i = 0; i < parameters_.n_components; ++i) {
    mu.col(i) =
        1.0 / n_samples_responsible(i) *
        static_cast<Vector<Dim>>((samples.transpose().array().colwise() *
                                  responsibilities.row(i).transpose().array())
                                     .transpose()
                                     .rowwise()
                                     .sum());
    const auto centered_samples =
        static_cast<StaticRowsMatrix<Dim>>(samples.colwise() - mu.col(i));
    const auto weighted_centered_samples = static_cast<StaticRowsMatrix<Dim>>(
        (centered_samples.transpose().array().colwise() *
         responsibilities.row(i).transpose().array())
            .transpose());
    for (size_t j = 0; j < n_samples; ++j) {
      sigma.block(0, i * Dim, Dim, Dim) += weighted_centered_samples.col(j) *
                                           centered_samples.transpose().row(j);
    }
    sigma.block(0, i * Dim, Dim, Dim) *= 1.0 / n_samples_responsible(i);
  }

  // Update Random Variables
  normal_covariance_scaling_ =
      parameters_.normal_prior_covariance_scaling + n_samples_responsible;
  for (size_t i = 0; i < parameters_.n_components; ++i) {
    normal_mean_.col(i) = 1.0 / normal_covariance_scaling_(i) *
                          (parameters_.normal_prior_covariance_scaling *
                               parameters_.normal_prior_mean +
                           n_samples_responsible(i) * mu.col(i));
    wishart_information_.block(0, i * Dim, Dim, Dim) =
        parameters_.wishart_prior_information +
        n_samples_responsible(i) * sigma.block(0, i * Dim, Dim, Dim) +
        parameters_.normal_prior_covariance_scaling * n_samples_responsible(i) /
            (parameters_.normal_prior_covariance_scaling +
             n_samples_responsible(i)) *
            (mu.col(i) - parameters_.normal_prior_mean) *
            (mu.col(i) - parameters_.normal_prior_mean).transpose();
    wishart_degrees_of_freedom_ =
        parameters_.wishart_prior_degrees_of_freedom + n_samples_responsible(i);
  }
  // TODO: After converging, assign the expectation of the random variables to
  // the parameters of the Gaussian mixture model
}

} // namespace gm

#endif // !GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_HPP
