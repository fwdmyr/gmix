#ifndef GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP
#define GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP

#include "BaseStrategy.hpp"
#include "Common.hpp"
#include "Statistics.hpp"
#include <limits>
#include <random>

namespace gm {

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
  MatrixX evaluate_responsibilities(const std::vector<GaussianComponent<Dim>> &,
                                    const StaticRowsMatrix<Dim> &) const;
  void estimate_parameters(std::vector<GaussianComponent<Dim>> &,
                           const StaticRowsMatrix<Dim> &,
                           const MatrixX &) const;
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

template <int Dim>
MatrixX ExpectationMaximizationStrategy<Dim>::evaluate_responsibilities(
    const std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) const {
  const auto n_samples = samples.cols();
  const auto n_components = parameters_.n_components;
  auto responsibilities = MatrixX::Zero(Dim, n_samples).eval();
  for (size_t i = 0; i < n_samples; ++i) {
    const Vector<Dim> sample = samples.col(i);
    auto responsibility = VectorX::Zero(parameters_.n_components).eval();
    for (size_t j = 0; j < parameters_.n_components; ++j) {
      auto &component = components[j];
      responsibility(j) = component(sample);
    }
    responsibilities.col(i) = responsibility;
  }
  responsibilities.colwise().normalize();
  return responsibilities;
}

template <int Dim>
void ExpectationMaximizationStrategy<Dim>::estimate_parameters(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples,
    const MatrixX &responsibilities) const {
  // TODO: Continue here. Prefer vectorized code!
  const auto n_samples = samples.cols();
  const auto n_samples_responsible = responsibilities.rowwise().sum();
  for (size_t i = 0; i < parameters_.n_components; ++i) {
    auto &component = components[i];
    component.set_weight(1.0 / n_samples * n_samples_responsible(i));
    const auto mu = 1.0 / n_samples_responsible(i) *
                    (samples.transpose().array().colwise() *
                     responsibilities.row(i).transpose().array())
                        .matrix()
                        .colwise()
                        .sum()
                        .eval();
    component.set_mean(mu);
    const auto centered_samples = samples.colwise() - mu;
  }
}

} // namespace gm

#endif // !#ifndef GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP
