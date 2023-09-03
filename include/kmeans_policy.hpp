#ifndef GMIX_K_MEANS_STRATEGY_HPP
#define GMIX_K_MEANS_STRATEGY_HPP

#include "kmeans_policy_impl.hpp"

namespace gmix {

template <int Dim> struct KMeansParameters {
  int n_components{0};
  int n_iterations{0};
  double early_stopping_threshold{0.0};
  bool warm_start{false};
};

template <int Dim> class KMeansPolicy {
public:
  using ParamType = KMeansParameters<Dim>;

  explicit KMeansPolicy(ParamType &&parameters) noexcept;

  explicit KMeansPolicy(const ParamType &parameters) noexcept;

  void fit(std::vector<GaussianComponent<Dim>> &,
           const StaticRowsMatrix<Dim> &) const;

  void initialize(std::vector<GaussianComponent<Dim>> &,
                  const StaticRowsMatrix<Dim> &) const;

protected:
  KMeansPolicy() = default;

private:
  ParamType parameters_{};
};

template <int Dim>
KMeansPolicy<Dim>::KMeansPolicy(ParamType &&parameters) noexcept
    : parameters_(std::move(parameters)) {}

template <int Dim>
KMeansPolicy<Dim>::KMeansPolicy(const ParamType &parameters) noexcept
    : parameters_(parameters) {}

template <int Dim>
void KMeansPolicy<Dim>::initialize(
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
void KMeansPolicy<Dim>::fit(std::vector<GaussianComponent<Dim>> &components,
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

#endif // !GMIX_K_MEANS_STRATEGY_HPP
