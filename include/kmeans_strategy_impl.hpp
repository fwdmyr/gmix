#ifndef GMIX_K_MEANS_STRATEGY_IMPL_HPP
#define GMIX_K_MEANS_STRATEGY_IMPL_HPP

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

} // namespace gmix

#endif // !GMIX_K_MEANS_STRATEGY_IMPL_HPP
