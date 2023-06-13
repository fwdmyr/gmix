#ifndef GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP
#define GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP

#include "BaseStrategy.hpp"
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
                   const std::vector<Vector<Dim>> &) const override;

private:
  std::vector<VectorX>
  evaluate_responsibilities(const std::vector<GaussianComponent<Dim>> &,
                            const std::vector<Vector<Dim>> &);
  void estimate_parameters(std::vector<GaussianComponent<Dim>> &,
                           const std::vector<Vector<Dim>> &,
                           const std::vector<VectorX> &);
  Parameters parameters_;
};

template <int Dim>
void ExpectationMaximizationStrategy<Dim>::fit(
    std::vector<GaussianComponent<Dim>> &components,
    const std::vector<Vector<Dim>> &samples) const {
  const auto n_components = parameters_.n_components;
  this->initialize(components, samples, n_components);
  for (size_t i = 0; i < parameters_.n_iterations; ++i) {
    const auto responsibilities =
        evaluate_responsibilities(components, samples);
    estimate_parameters(components, samples, responsibilities);
  }
}

template <int Dim>
std::vector<VectorX>
ExpectationMaximizationStrategy<Dim>::evaluate_responsibilities(
    const std::vector<GaussianComponent<Dim>> &components,
    const std::vector<Vector<Dim>> &samples) {
  const auto n_samples = samples.size();
  std::vector<VectorX> responsibilities;
  responsibilities.reserve(n_samples);
  for (size_t i = 0; i < n_samples; ++i) {
    const auto sample = samples[i];
    auto responsibility = VectorX::Zero(parameters_.n_components);
    for (size_t j = 0; j < parameters_.n_components; ++j) {
      auto &component = components[j];
      responsibility(j) = component(sample);
    }
    const auto normalizer = responsibility.sum();
    responsibility = 1.0 / normalizer * responsibility;
    responsibilities.push_back(responsibility);
  }
  return {};
}

template <int Dim>
void ExpectationMaximizationStrategy<Dim>::estimate_parameters(
    std::vector<GaussianComponent<Dim>> &components,
    const std::vector<Vector<Dim>> &samples,
    const std::vector<VectorX> &responsibilities) {
  return;
}

} // namespace gm

#endif // !#ifndef GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP
