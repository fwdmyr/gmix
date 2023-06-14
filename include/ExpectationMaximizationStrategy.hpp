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
  std::vector<double>
  evaluate_responsibilities(const std::vector<GaussianComponent<Dim>> &,
                            const StaticRowsMatrix<Dim> &);
  void estimate_parameters(std::vector<GaussianComponent<Dim>> &,
                           const StaticRowsMatrix<Dim> &,
                           const std::vector<double> &);
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
std::vector<double>
ExpectationMaximizationStrategy<Dim>::evaluate_responsibilities(
    const std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) {
  return {};
}

template <int Dim>
void ExpectationMaximizationStrategy<Dim>::estimate_parameters(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples,
    const std::vector<double> &responsibilities) {
  return;
}

} // namespace gm

#endif // !#ifndef GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP
