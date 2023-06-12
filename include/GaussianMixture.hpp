#ifndef GMSAM_GAUSSIAN_MIXTURE_HPP
#define GMSAM_GAUSSIAN_MIXTURE_HPP

#include "FittingStrategy.hpp"
#include <initializer_list>
#include <random>
#include <vector>

namespace gm {

template <int Dim> class GaussianMixture {

public:
  using container_type = std::vector<GaussianComponent<Dim>>;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;

  GaussianMixture() : strategy_(NoneStrategy<Dim>{}){};
  GaussianMixture(std::initializer_list<GaussianComponent<Dim>>);
  GaussianMixture(const GaussianMixture<Dim> &) = default;
  GaussianMixture(GaussianMixture<Dim> &&) = default;
  GaussianMixture<Dim> &operator=(const GaussianMixture<Dim> &) = default;
  GaussianMixture<Dim> &operator=(GaussianMixture<Dim> &&) = default;

  inline iterator begin() noexcept { return components_.begin(); }
  inline const_iterator cbegin() const noexcept { return components_.cbegin(); }
  inline iterator end() noexcept { return components_.end(); }
  inline const_iterator cend() const noexcept { return components_.cend(); }

  const GaussianComponent<Dim> &operator[](size_t idx) const {
    return components_[idx];
  }
  size_t get_size() const { return components_.size(); }

  void add_component(const GaussianComponent<Dim> &component) {
    components_.push_back(component);
  };
  void add_component(GaussianComponent<Dim> &&component) {
    components_.emplace_back(component);
  }

  void reset() { components_.clear(); }

  void set_strategy(const FittingStrategy<Dim> &strategy) {
    strategy_ = strategy;
  }
  void set_strategy(FittingStrategy<Dim> &&strategy) { strategy_ = strategy; }

  void fit(const std::vector<Vector<Dim>> &samples) {
    strategy_.fit(components_, samples);
  }

private:
  std::vector<GaussianComponent<Dim>> components_;
  FittingStrategy<Dim> strategy_;
};

template <int Dim>
std::vector<Vector<Dim>>
draw_from_gaussian_mixture(const GaussianMixture<Dim> &gmm, size_t n_samples) {

  static std::mt19937 gen{std::random_device{}()};
  static std::normal_distribution<> nd;

  std::vector<Vector<Dim>> samples;
  samples.reserve(n_samples);

  std::vector<double> weights;
  weights.reserve(gmm.get_size());
  for (auto it = gmm.cbegin(); it < gmm.cend(); ++it)
    weights.push_back(it->get_weight());
  std::discrete_distribution<> dd(weights.begin(), weights.end());

  for (size_t i = 0; i < n_samples; ++i) {
    const auto &component = gmm[dd(gen)];
    Eigen::SelfAdjointEigenSolver<Matrix<Dim, Dim>> eigen_solver(
        component.get_covariance());
    const auto transform = eigen_solver.eigenvectors() *
                           eigen_solver.eigenvalues().cwiseSqrt().asDiagonal();
    const auto sample =
        component.get_mean() +
        transform * Vector<Dim>{}.unaryExpr([](auto x) { return nd(gen); });
    samples.push_back(sample);
  }
  return samples;
}

} // namespace gm
#endif // !GMSAM_GAUSSIAN_MIXTURE_HPP
