#ifndef GMSAM_K_MEANS_STRATEGY_HPP
#define GMSAM_K_MEANS_STRATEGY_HPP

#include "BaseStrategy.hpp"
#include "Statistics.hpp"
#include <limits>
#include <numeric>
#include <random>

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

template <int Dim>
std::vector<std::vector<Vector<Dim>>> partition_samples_responsibly(
    const std::vector<GaussianComponent<Dim>> &components,
    const std::vector<Vector<Dim>> &samples) {
  const auto n_components = components.size();
  std::vector<std::vector<Vector<Dim>>> partitions{n_components};
  for (const auto &sample : samples) {
    auto squared_l2_min = std::numeric_limits<double>::max();
    int responsibility = -1;
    for (size_t i = 0; i < n_components; ++i) {
      const auto &component = components[i];
      const auto squared_l2 = (sample - component.get_mean()).norm();
      if (squared_l2 <= squared_l2_min) {
        squared_l2_min = squared_l2;
        responsibility = i;
      }
    }
    partitions[responsibility].push_back(sample);
  }
  return partitions;
}

} // namespace

template <int Dim> class KMeansStrategy final : public BaseStrategy<Dim> {
public:
  struct Parameters {
    int n_components;
    int n_iterations;
  };

  explicit KMeansStrategy(const Parameters &parameters)
      : parameters_(parameters) {}
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const std::vector<Vector<Dim>> &) const override;

private:
  void initialize(std::vector<GaussianComponent<Dim>> &,
                  const std::vector<Vector<Dim>> &) const;
  void update_weight(std::vector<GaussianComponent<Dim>> &,
                     const std::vector<std::vector<Vector<Dim>>> &) const;
  void update_mean(std::vector<GaussianComponent<Dim>> &,
                   const std::vector<std::vector<Vector<Dim>>> &) const;
  void update_covariance(std::vector<GaussianComponent<Dim>> &,
                         const std::vector<std::vector<Vector<Dim>>> &) const;

  Parameters parameters_;
};

template <int Dim>
void KMeansStrategy<Dim>::fit(std::vector<GaussianComponent<Dim>> &components,
                              const std::vector<Vector<Dim>> &samples) const {
  initialize(components, samples);
  std::vector<std::vector<Vector<Dim>>> partitions;
  for (size_t i = 0; i < parameters_.n_iterations; ++i) {
    partitions = partition_samples_responsibly(components, samples);
    update_mean(components, partitions);
  }
  update_weight(components, partitions);
  update_covariance(components, partitions);
}

template <int Dim>
void KMeansStrategy<Dim>::initialize(
    std::vector<GaussianComponent<Dim>> &components,
    const std::vector<Vector<Dim>> &samples) const {
  const auto n_components = parameters_.n_components;
  components.resize(0);
  const auto partitions = partition_samples_randomly(samples, n_components);
  for (const auto &partition : partitions) {
    const auto mu = sample_mean(partition);
    const auto sigma = sample_covariance(partition, mu);
    components.push_back({1.0 / n_components, mu, sigma});
  }
}

template <int Dim>
void KMeansStrategy<Dim>::update_weight(
    std::vector<GaussianComponent<Dim>> &components,
    const std::vector<std::vector<Vector<Dim>>> &partitions) const {
  const auto n_samples = std::accumulate(
      partitions.begin(), partitions.end(), 0,
      [](auto acc, const auto &rhs) { return acc + rhs.size(); });
  const auto n_components = parameters_.n_components;
  for (size_t i = 0; i < n_components; ++i) {
    auto &component = components[i];
    auto &partition = partitions[i];
    const auto weight = static_cast<double>(partition.size()) / n_samples;
    component.set_weight(weight);
  }
}

template <int Dim>
void KMeansStrategy<Dim>::update_mean(
    std::vector<GaussianComponent<Dim>> &components,
    const std::vector<std::vector<Vector<Dim>>> &partitions) const {
  const auto n_components = parameters_.n_components;
  for (size_t i = 0; i < n_components; ++i) {
    auto &component = components[i];
    const auto mu = sample_mean(partitions[i]);
    component.set_mean(mu);
  }
}

template <int Dim>
void KMeansStrategy<Dim>::update_covariance(
    std::vector<GaussianComponent<Dim>> &components,
    const std::vector<std::vector<Vector<Dim>>> &partitions) const {
  const auto n_components = parameters_.n_components;
  for (size_t i = 0; i < n_components; ++i) {
    auto &component = components[i];
    const auto mu = component.get_mean();
    const auto sigma = sample_covariance(partitions[i], mu);
    component.set_covariance(sigma);
  }
}

} // namespace gm
#endif // !GMSAM_K_MEANS_STRATEGY_HPP
