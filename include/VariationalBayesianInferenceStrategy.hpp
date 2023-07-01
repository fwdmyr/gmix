#ifndef GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_HPP
#define GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_HPP

#include "BaseStrategy.hpp"
#include "Common.hpp"
#include "KMeansStrategy.hpp"
#include "Statistics.hpp"
#include <eigen3/Eigen/src/Core/GlobalFunctions.h>
#include <limits>
#include <random>
#include <unsupported/Eigen/SpecialFunctions>
#include <variant>

namespace gm {

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
      : parameters_(parameters) {}
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const StaticRowsMatrix<Dim> &) const override;

private:
  Parameters parameters_{};

  void initialize(std::vector<GaussianComponent<Dim>> &,
                  const StaticRowsMatrix<Dim> &) const;
  void iterate(const StaticRowsMatrix<Dim> &);
};

namespace {

template <int Dim>
using Parameters =
    typename VariationalBayesianInferenceStrategy<Dim>::Parameters;

template <int Dim>
void evaluate_responsibilities(
    const VectorX &dirichlet_weight, const StaticRowsMatrix<Dim> &normal_mean,
    const VectorX &normal_covariance_scaling,
    const StaticRowsMatrix<Dim> &&wishart_information,
    const VectorX &wishart_degrees_of_freedom,
    const StaticRowsMatrix<Dim> &samples, const Parameters<Dim> &parameters,
    MatrixX &responsibilities) {
  const auto n_samples = samples.cols();
  for (size_t i = 0; i < parameters.n_components; ++i) {
    const auto e_logweight = Eigen::numext::digamma(dirichlet_weight(i)) -
                             Eigen::numext::digamma(dirichlet_weight.sum());
    auto e_logdetcovariance =
        Dim * std::log(2) +
        std::log(1.0 /
                 wishart_information.block(0, i * Dim, Dim, Dim).determinant());
    for (size_t d = 1; d <= Dim; ++d) {
      e_logdetcovariance +=
          Eigen::numext::digamma(0.5 * (wishart_degrees_of_freedom(i) + 1 - d));
    }
    for (size_t j = 0; j < n_samples; ++j) {
      const auto distance = samples.col(j) - normal_mean.col(i);
      const auto e_squaredmahalanobisdistance =
          static_cast<double>(Dim) / normal_covariance_scaling(i) +
          wishart_degrees_of_freedom(i) * distance.transpose() *
              wishart_information.block(0, i * Dim, Dim, Dim).inverse() *
              distance;
      const auto logresponsibility = e_logweight + 0.5 * e_logdetcovariance -
                                     0.5 * Dim * std::log(2.0 * M_PI) -
                                     0.5 * e_squaredmahalanobisdistance;
      responsibilities(i, j) = std::exp(logresponsibility);
    }
  }
  responsibilities.colwise().normalize();
}

template <int Dim>
void compute_statistics(const StaticRowsMatrix<Dim> &responsibilities,
                        const StaticRowsMatrix<Dim> &samples,
                        const Parameters<Dim> &parameters,
                        VectorX &n_samples_responsible,
                        StaticRowsMatrix<Dim> &mu,
                        StaticRowsMatrix<Dim> &sigma) {
  const auto n_samples = samples.cols();
  for (size_t i = 0; i < parameters.n_components; ++i) {
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
}

template <int Dim>
void update_random_variables(const VectorX &n_samples_responsible,
                             const StaticRowsMatrix<Dim> &mu,
                             const StaticRowsMatrix<Dim> &sigma,
                             const Parameters<Dim> &parameters,
                             VectorX &dirichlet_weight,
                             StaticRowsMatrix<Dim> &normal_mean,
                             VectorX &normal_covariance_scaling,
                             StaticRowsMatrix<Dim> &&wishart_information,
                             VectorX &wishart_degrees_of_freedom) {

  normal_covariance_scaling =
      parameters.normal_prior_covariance_scaling + n_samples_responsible;
  for (size_t i = 0; i < parameters.n_components; ++i) {
    normal_mean.col(i) = 1.0 / normal_covariance_scaling(i) *
                         (parameters.normal_prior_covariance_scaling *
                              parameters.normal_prior_mean +
                          n_samples_responsible(i) * mu.col(i));
    wishart_information.block(0, i * Dim, Dim, Dim) =
        parameters.wishart_prior_information +
        n_samples_responsible(i) * sigma.block(0, i * Dim, Dim, Dim) +
        parameters.normal_prior_covariance_scaling * n_samples_responsible(i) /
            (parameters.normal_prior_covariance_scaling +
             n_samples_responsible(i)) *
            (mu.col(i) - parameters.normal_prior_mean) *
            (mu.col(i) - parameters.normal_prior_mean).transpose();
    wishart_degrees_of_freedom =
        parameters.wishart_prior_degrees_of_freedom + n_samples_responsible(i);
  }
}

} // namespace

template <int Dim>
void VariationalBayesianInferenceStrategy<Dim>::initialize(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) const {
  const typename KMeansStrategy<Dim>::Parameters initialization_parameters = {
      parameters_.n_components, 1, parameters_.warm_start};
  const auto initialization_strategy =
      KMeansStrategy<Dim>{initialization_parameters};
  initialization_strategy.fit(components, samples);
}

template <int Dim>
void VariationalBayesianInferenceStrategy<Dim>::fit(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) const {

  const auto n_samples = samples.cols();
  const auto n_components = parameters_.n_components;

  auto dirichlet_weight = static_cast<VectorX>(
      parameters_.dirichlet_prior_weight.replicate(1, n_components));
  auto normal_mean = static_cast<StaticRowsMatrix<Dim>>(
      parameters_.normal_prior_mean.replicate(1, n_components));
  auto normal_covariance_scaling = static_cast<VectorX>(
      parameters_.normal_prior_covariance_scaling.replicate(1, n_components));
  auto wishart_information = static_cast<StaticRowsMatrix<Dim>>(
      parameters_.wishart_prior_information.replicate(1, n_components));
  auto wishart_degrees_of_freedom = static_cast<VectorX>(
      parameters_.wishart_prior_degrees_of_freedom.replicate(1, n_components));

  if (!parameters_.warm_start || components.size() != n_components)
    initialize(components, samples);

  for (size_t i = 0; i < n_components; ++i) {
    const auto &component = components[i];
    dirichlet_weight(i) = component.get_weight();
    normal_mean.col(i) = component.get_mean();
    wishart_information.block(0, i * Dim, Dim, Dim) =
        component.get_sqrt_information().transpose() *
        component.get_sqrt_information();
  }

  auto responsibilities = MatrixX{n_components, n_samples};
  auto n_samples_responsible = VectorX{n_components, 1};
  auto mu = StaticRowsMatrix<Dim>{Dim, n_components};
  auto sigma = StaticRowsMatrix<Dim>{Dim, Dim * n_components};

  for (size_t i = 0; i < parameters_.n_iterations; ++i) {

    evaluate_responsibilities(dirichlet_weight, normal_mean,
                              normal_covariance_scaling, wishart_information,
                              wishart_degrees_of_freedom, samples, parameters_,
                              responsibilities);

    compute_statistics(responsibilities, samples, parameters_,
                       n_samples_responsible, mu, sigma);

    update_random_variables(n_samples_responsible, mu, sigma, parameters_,
                            dirichlet_weight, normal_mean,
                            normal_covariance_scaling, wishart_information,
                            wishart_degrees_of_freedom);
  }

  for (size_t i = 0; i < n_components; ++i) {
    const auto &component = components[i];
    component.set_weight(dirichlet_weight(i));
    component.set_mean(normal_mean);
    component.set_covariance(wishart_information.inverse());
  }
}

} // namespace gm

#endif // !GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_HPP
