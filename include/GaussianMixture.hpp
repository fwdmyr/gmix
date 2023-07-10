#ifndef GMSAM_GAUSSIAN_MIXTURE_HPP
#define GMSAM_GAUSSIAN_MIXTURE_HPP

#include "GaussianComponent.hpp"
#include "TypeTraits.hpp"
#include <initializer_list>
#include <memory>
#include <random>
#include <vector>

namespace gmix {

template <int Dim> class GaussianMixture {

public:
  using container_type = std::vector<GaussianComponent<Dim>>;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;

  GaussianMixture() = default;
  GaussianMixture(std::initializer_list<GaussianComponent<Dim>>);
  GaussianMixture(const GaussianMixture<Dim> &) = default;
  GaussianMixture(GaussianMixture<Dim> &&) = default;
  GaussianMixture<Dim> &operator=(const GaussianMixture<Dim> &) = default;
  GaussianMixture<Dim> &operator=(GaussianMixture<Dim> &&) = default;

  [[nodiscard]] inline iterator begin() noexcept { return components_.begin(); }
  [[nodiscard]] inline const_iterator cbegin() const noexcept {
    return components_.cbegin();
  }
  [[nodiscard]] inline iterator end() noexcept { return components_.end(); }
  [[nodiscard]] inline const_iterator cend() const noexcept {
    return components_.cend();
  }

  [[nodiscard]] const GaussianComponent<Dim> &get_component(size_t idx) const {
    return components_[idx];
  }

  [[nodiscard]] size_t get_size() const { return components_.size(); }

  [[nodiscard]] const std::vector<GaussianComponent<Dim>> &
  get_components() const {
    return components_;
  }

  [[nodiscard]] std::vector<GaussianComponent<Dim>> &get_components() {
    return components_;
  }

  void add_component() { components_.emplace_back(); }

  void add_component(const GaussianComponent<Dim> &component) {
    components_.push_back(component);
  };

  void add_component(GaussianComponent<Dim> &&component) {
    components_.emplace_back(component);
  }

  void reset() { components_.resize(0); }

  friend std::ostream &operator<<(std::ostream &os,
                                  const GaussianMixture<Dim> &gmm) {
    os << "GaussianMixture<" << Dim << ">\n";
    for (const auto &component : gmm.components_)
      os << component << '\n';
    return os;
  }

private:
  std::vector<GaussianComponent<Dim>> components_{};
};

template <typename StrategyType, typename MatrixType,
          typename GaussianMixtureType>
void fit(const MatrixType &samples, const StrategyType &strategy,
         GaussianMixtureType &gmm) {
  auto &components = gmm.get_components();
  strategy.fit(components, samples);
}

template <int Dim>
[[nodiscard]] StaticRowsMatrix<Dim>
draw_from_gaussian_mixture(const GaussianMixture<Dim> &gmm, size_t n_samples) {
  static std::mt19937 gen{std::random_device{}()};
  static std::normal_distribution<> nd;

  std::vector<double> weights;
  weights.reserve(gmm.get_size());
  for (auto it = gmm.cbegin(); it < gmm.cend(); ++it)
    weights.push_back(it->get_weight());
  std::discrete_distribution<> dd(weights.begin(), weights.end());
  auto samples = StaticRowsMatrix<Dim>::Zero(Dim, n_samples).eval();

  for (size_t i = 0; i < n_samples; ++i) {
    const auto &component = gmm.get_component(dd(gen));
    Eigen::SelfAdjointEigenSolver<Matrix<Dim, Dim>> eigen_solver(
        component.get_covariance());
    const auto transform = eigen_solver.eigenvectors() *
                           eigen_solver.eigenvalues().cwiseSqrt().asDiagonal();
    samples.col(i) =
        component.get_mean() +
        transform * ColVector<Dim>{}.unaryExpr([](auto x) { return nd(gen); });
  }
  return samples;
}

} // namespace gmix
#endif // !GMSAM_GAUSSIAN_MIXTURE_HPP
