#ifndef GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP
#define GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP

#include "BaseStrategy.hpp"
#include "Common.hpp"
#include "Statistics.hpp"
#include <limits>
#include <random>

namespace gm {

namespace {

template <int Dim>
MatrixX
evaluate_responsibilities(const std::vector<GaussianComponent<Dim>> &components,
                          const StaticRowsMatrix<Dim> &samples) {
  const auto n_samples = samples.cols();
  const auto n_components = components.size();
  auto responsibilities = static_cast<StaticRowsMatrix<Dim>>(
      StaticRowsMatrix<Dim>::Zero(Dim, n_samples));
  for (size_t i = 0; i < n_samples; ++i) {
    const auto sample = static_cast<Vector<Dim>>(samples.col(i));
    auto responsibility = static_cast<VectorX>(VectorX::Zero(n_components));
    for (size_t j = 0; j < n_components; ++j) {
      auto &component = components[j];
      responsibility(j) = component(sample);
    }
    responsibilities.col(i) = responsibility;
  }
  responsibilities.colwise().normalize();
  return responsibilities;
}

template <int Dim>
void estimate_parameters(std::vector<GaussianComponent<Dim>> &components,
                         const StaticRowsMatrix<Dim> &samples,
                         const MatrixX &responsibilities) {
  const auto n_samples = samples.cols();
  const auto n_samples_responsible =
      static_cast<VectorX>(responsibilities.rowwise().sum());
  for (size_t i = 0; i < components.size(); ++i) {
    auto &component = components[i];
    component.set_weight(1.0 / n_samples * n_samples_responsible(i));
    const auto mu =
        1.0 / n_samples_responsible(i) *
        static_cast<Vector<Dim>>((samples.transpose().array().colwise() *
                                  responsibilities.row(i).transpose().array())
                                     .transpose()
                                     .rowwise()
                                     .sum());
    component.set_mean(mu);
    const auto centered_samples = static_cast<StaticRowsMatrix<Dim>>(
        samples.colwise() - component.get_mean());
    const auto weighted_centered_samples = static_cast<StaticRowsMatrix<Dim>>(
        (samples.transpose().array().colwise() *
         responsibilities.row(i).transpose().array())
            .transpose());
    // TODO: Can Eigen's reductions handle the sum over the outer products?
    // Maybe user-defined redux?
    auto sigma = static_cast<Matrix<Dim, Dim>>(Matrix<Dim, Dim>::Zero().eval());
    for (size_t j = 0; j < n_samples; ++j) {
      sigma += weighted_centered_samples.col(j) *
               centered_samples.transpose().row(j);
    }
    sigma *= 1.0 / n_samples_responsible(i);
    component.set_covariance(sigma);
  }
}

} // namespace

template <int Dim>
class ExpectationMaximizationStrategy final : public BaseStrategy<Dim> {
public:
  struct Parameters {
    int n_components;
    int n_iterations;
  };

  explicit ExpectationMaximizationStrategy(const Parameters &parameters)
      : parameters_(parameters) {}
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const StaticRowsMatrix<Dim> &) const override;

private:
  Parameters parameters_;
};

template <int Dim>
void ExpectationMaximizationStrategy<Dim>::fit(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) const {
  const auto n_components = parameters_.n_components;
  this->initialize(components, samples, n_components);
  for (size_t i = 0; i < parameters_.n_iterations; ++i) {
    const auto responsibilities =
        evaluate_responsibilities(components, samples);
    estimate_parameters(components, samples, responsibilities);
  }
}

} // namespace gm

#endif // !#ifndef GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP
