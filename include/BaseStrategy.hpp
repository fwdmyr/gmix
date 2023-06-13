#ifndef GMSAM_BASE_STRATEGY_HPP
#define GMSAM_BASE_STRATEGY_HPP

#include "GaussianComponent.hpp"
#include "Statistics.hpp"
#include <stdexcept>
#include <vector>

namespace gm {

namespace {

template <int Dim>
std::vector<std::vector<Vector<Dim>>>
partition_samples_randomly(const std::vector<Vector<Dim>> &samples,
                           size_t n_partitions) {
  static std::mt19937 gen{std::random_device{}()};

  std::vector<std::vector<Vector<Dim>>> partitions{n_partitions};
  std::vector<double> discrete_probabilities(n_partitions, 1.0 / n_partitions);
  std::discrete_distribution<> dd(discrete_probabilities.begin(),
                                  discrete_probabilities.end());
  for (const auto &sample : samples)
    partitions[dd(gen)].push_back(sample);
  return partitions;
}

} // namespace

template <int Dim> class BaseStrategy {
public:
  virtual ~BaseStrategy() = default;
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const std::vector<Vector<Dim>> &) const = 0;

protected:
  void initialize(std::vector<GaussianComponent<Dim>> &,
                  const std::vector<Vector<Dim>> &, int) const;
};

template <int Dim>
void BaseStrategy<Dim>::initialize(
    std::vector<GaussianComponent<Dim>> &components,
    const std::vector<Vector<Dim>> &samples, int n_components) const {
  components.resize(0);
  const auto partitions = partition_samples_randomly(samples, n_components);
  for (const auto &partition : partitions) {
    const auto mu = sample_mean(partition);
    const auto sigma = sample_covariance(partition, mu);
    components.push_back({1.0 / n_components, mu, sigma});
  }
}

} // namespace gm
#endif // !GMSAM_BASE_STRATEGY_HPP
