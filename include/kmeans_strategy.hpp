#ifndef GMSAM_K_MEANS_STRATEGY_HPP
#define GMSAM_K_MEANS_STRATEGY_HPP

#include "common.hpp"
#include "gaussian_component.hpp"
#include "statistics.hpp"
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace gmix {

template <int Dim> struct KMeansParameters;
template <int Dim> class KMeansStrategy;

namespace internal {

template <int Dim>
[[nodiscard]] std::vector<StaticRowsMatrix<Dim>>
partition_samples(const StaticRowsMatrix<Dim> &samples, size_t n_partitions) {
  if (n_partitions == 0)
    return {};
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

template <int Dim>
[[nodiscard]] std::vector<StaticRowsMatrix<Dim>> partition_samples_responsibly(
    const std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) {
  assert(!components.empty());
  assert(samples.cols() >= components.size());
  const auto n_components = components.size();
  std::vector<std::vector<int>> responsibilities{n_components};
  std::for_each(
      responsibilities.begin(), responsibilities.end(),
      [n_samples = samples.cols()](auto &component_responsibilities) -> void {
        component_responsibilities.reserve(n_samples);
      });
  // TODO: Probably also not vectorizable as we need to access the components
  for (size_t i = 0; i < samples.cols(); ++i) {
    auto squared_l2_min = std::numeric_limits<double>::max();
    size_t dominant_mode = 0;
    for (size_t j = 0; j < n_components; ++j) {
      const auto &component = components[j];
      const auto squared_l2 =
          (samples.col(i) - component.get_mean()).squaredNorm();
      if (squared_l2 <= squared_l2_min) {
        squared_l2_min = squared_l2;
        dominant_mode = j;
      }
    }
    responsibilities[dominant_mode].push_back(i);
  }
  std::vector<StaticRowsMatrix<Dim>> partitions;
  partitions.reserve(n_components);
  for (size_t i = 0; i < n_components; ++i) {
    const auto partition_size = responsibilities[i].size();
    auto partition = static_cast<StaticRowsMatrix<Dim>>(
        StaticRowsMatrix<Dim>::Zero(Dim, partition_size));
    for (size_t j = 0; j < partition_size; ++j) {
      partition.col(j) = samples.col(responsibilities[i][j]);
    }
    partitions.push_back(partition);
  }
  return partitions;
}

template <int Dim>
void update_weight(std::vector<GaussianComponent<Dim>> &components,
                   const std::vector<StaticRowsMatrix<Dim>> &partitions) {
  assert(components.size() == partitions.size());
  const auto n_samples = std::accumulate(
      partitions.begin(), partitions.end(), 0,
      [](auto acc, const auto &rhs) { return acc + rhs.cols(); });
  const auto n_components = components.size();
  for (size_t i = 0; i < n_components; ++i) {
    auto &component = components[i];
    const auto &partition = partitions[i];
    const auto weight = static_cast<double>(partition.cols()) / n_samples;
    component.set_weight(weight);
  }
}

template <int Dim>
void update_mean(std::vector<GaussianComponent<Dim>> &components,
                 const std::vector<StaticRowsMatrix<Dim>> &partitions) {
  assert(components.size() == partitions.size());
  const auto n_components = components.size();
  for (size_t i = 0; i < n_components; ++i) {
    auto &component = components[i];
    const auto mu = sample_mean(partitions[i]);
    component.set_mean(mu);
  }
}

template <int Dim>
void update_covariance(std::vector<GaussianComponent<Dim>> &components,
                       const std::vector<StaticRowsMatrix<Dim>> &partitions) {
  assert(components.size() == partitions.size());
  const auto n_components = components.size();
  for (size_t i = 0; i < n_components; ++i) {
    auto &component = components[i];
    const auto sigma = sample_covariance(partitions[i]);
    component.set_covariance(sigma);
  }
}

template <int Dim>
[[nodiscard]] StaticRowsMatrix<Dim>
get_mean_matrix(const std::vector<GaussianComponent<Dim>> &components) {
  assert(!components.empty());
  const auto n_components = components.size();
  auto mean_matrix = static_cast<StaticRowsMatrix<Dim>>(
      StaticRowsMatrix<Dim>::Zero(Dim, n_components));
  for (size_t i = 0; i < n_components; ++i) {
    const auto &component = components[i];
    mean_matrix.col(i) = component.get_mean();
  }
  return mean_matrix;
}
template <int Dim>
[[nodiscard]] bool is_early_stopping_condition_fulfilled(
    const StaticRowsMatrix<Dim> &current_mean_matrix,
    const StaticRowsMatrix<Dim> &new_mean_matrix,
    double early_stopping_threshold) {
  assert(current_mean_matrix.cols() == new_mean_matrix.cols());
  const auto squared_norms =
      (current_mean_matrix - new_mean_matrix).colwise().squaredNorm();
  return squared_norms.maxCoeff() <
         early_stopping_threshold * early_stopping_threshold;
}

} // namespace internal

template <int Dim> struct KMeansParameters {
  int n_components{0};
  int n_iterations{0};
  double early_stopping_threshold{0.0};
  bool warm_start{false};
};

template <int Dim> class KMeansStrategy {
public:
  using ParamType = KMeansParameters<Dim>;

  explicit KMeansStrategy(ParamType &&parameters) noexcept
      : parameters_(std::move(parameters)) {}

  explicit KMeansStrategy(const ParamType &parameters) noexcept
      : parameters_(parameters) {}

  void fit(std::vector<GaussianComponent<Dim>> &,
           const StaticRowsMatrix<Dim> &) const;

  void initialize(std::vector<GaussianComponent<Dim>> &,
                  const StaticRowsMatrix<Dim> &) const;

protected:
  KMeansStrategy() = default;

private:
  ParamType parameters_{};
};

template <int Dim>
void KMeansStrategy<Dim>::initialize(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) const {
  assert(samples.cols() >= parameters_.n_components);
  components.resize(0);
  const auto partitions =
      internal::partition_samples(samples, parameters_.n_components);
  for (const auto &partition : partitions) {
    const auto mu = internal::sample_mean(partition);
    const auto sigma = internal::sample_covariance(partition, mu);
    components.push_back({1.0 / parameters_.n_components, mu, sigma});
  }
}

template <int Dim>
void KMeansStrategy<Dim>::fit(std::vector<GaussianComponent<Dim>> &components,
                              const StaticRowsMatrix<Dim> &samples) const {
  assert(samples.cols() >= parameters_.n_components);
  const auto n_components = parameters_.n_components;
  if (!parameters_.warm_start || components.size() != n_components)
    initialize(components, samples);
  std::vector<StaticRowsMatrix<Dim>> partitions;
  auto current_mean_matrix = internal::get_mean_matrix(components);
  for (size_t i = 0; i < parameters_.n_iterations; ++i) {
    partitions = internal::partition_samples_responsibly(components, samples);
    internal::update_mean(components, partitions);
    const auto new_mean_matrix = internal::get_mean_matrix(components);
    if (internal::is_early_stopping_condition_fulfilled(
            current_mean_matrix, new_mean_matrix,
            parameters_.early_stopping_threshold))
      break;
    else
      current_mean_matrix = new_mean_matrix;
  }
  internal::update_weight(components, partitions);
  internal::update_covariance(components, partitions);
}

} // namespace gmix
#endif // !GMSAM_K_MEANS_STRATEGY_HPP
