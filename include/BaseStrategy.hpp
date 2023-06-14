#ifndef GMSAM_BASE_STRATEGY_HPP
#define GMSAM_BASE_STRATEGY_HPP

#include "Common.hpp"
#include "GaussianComponent.hpp"
#include "Statistics.hpp"
#include <stdexcept>
#include <vector>

namespace gm {

namespace {

template <int Dim>
std::vector<StaticRowsMatrix<Dim>>
partition_samples_randomly(const StaticRowsMatrix<Dim> &samples,
                           size_t n_partitions) {
  const auto n_samples = samples.cols();
  std::vector<StaticRowsMatrix<Dim>> partitions;
  partitions.reserve(n_partitions);
  const auto partition_size =
      static_cast<int>(static_cast<double>(n_samples) / n_partitions);
  for (size_t left_index = 0; left_index < n_samples;
       left_index += partition_size) {
    const auto batch_size =
        std::min<size_t>(partition_size, n_samples - left_index);
    partitions.push_back(samples.middleCols(left_index, batch_size));
  }
  return partitions;
}

} // namespace

template <int Dim> class BaseStrategy {
public:
  virtual ~BaseStrategy() = default;
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const StaticRowsMatrix<Dim> &) const = 0;

protected:
  void initialize(std::vector<GaussianComponent<Dim>> &,
                  const StaticRowsMatrix<Dim> &, size_t) const;
};

template <int Dim>
void BaseStrategy<Dim>::initialize(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples, size_t n_components) const {
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
