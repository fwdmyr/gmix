#ifndef GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_HPP
#define GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_HPP

#include "BaseStrategy.hpp"
#include "Common.hpp"
#include "Statistics.hpp"
#include <limits>
#include <random>

namespace gm {

namespace {

template <int Dim> class VariationalBayesianInferenceStrategy {
public:
  struct Parameters;
};

template <int Dim>
MatrixX evaluate_responsibilities(
    const std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples,
    const typename VariationalBayesianInferenceStrategy<Dim>::Parameters
        &parameters) {
  const auto n_samples = samples.cols();
  const auto n_components = components.size();
  auto responsibilities = static_cast<MatrixX>(MatrixX::Zero(Dim, n_samples));
  for (size_t i = 0; i < n_samples; ++i) {
    const auto sample = static_cast<Vector<Dim>>(samples.col(i));
    auto responsibility = static_cast<VectorX>(VectorX::Zero(n_components));
    for (size_t j = 0; j < n_components; ++j) {
      auto &component = components[j];
      responsibility(j) = component(sample);
    }
    responsibilities.col(i) = 1.0 / responsibility.sum() * responsibility;
  }
  return responsibilities;
}

} // namespace

template <int Dim>
class VariationalBayesianInferenceStrategy final : public BaseStrategy<Dim> {
public:
  struct Parameters {
    int n_components;
    int n_iterations;
    Vector<Dim> normal_prior_mean;
    Matrix<Dim, Dim> normal_prior_covariance;
    double wishart_prior_degrees_of_freedom;
    Matrix<Dim, Dim> wishart_prior_covariance;
  };

  explicit VariationalBayesianInferenceStrategy(const Parameters &parameters)
      : parameters_(parameters) {}
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const StaticRowsMatrix<Dim> &) const override;

private:
  Parameters parameters_;
};

} // namespace gm

#endif // !GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_HPP
