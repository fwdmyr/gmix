#ifndef GMSAM_GAUSSIAN_MIXTURE_HPP
#define GMSAM_GAUSSIAN_MIXTURE_HPP

#include "gaussian_component.hpp"
#include "null_strategy.hpp"
#include "statistics.hpp"
#include <memory>
#include <random>
#include <vector>

namespace gmix {

template <int Dim, template <int> typename FittingStrategy = NullStrategy>
class GaussianMixture : public FittingStrategy<Dim> {

public:
  explicit GaussianMixture() noexcept;

  explicit GaussianMixture(
      std::initializer_list<GaussianComponent<Dim>>) noexcept;

  template <typename Parameters,
            typename = std::enable_if_t<is_constructible_with_v<
                FittingStrategy<Dim>, std::decay_t<Parameters>>>>
  explicit GaussianMixture(Parameters &&) noexcept;

  template <typename Parameters,
            typename = std::enable_if_t<is_constructible_with_v<
                FittingStrategy<Dim>, std::decay_t<Parameters>>>>
  GaussianMixture(std::initializer_list<GaussianComponent<Dim>>,
                  Parameters &&) noexcept;

  [[nodiscard]] const auto &get_component(size_t idx) const;

  [[nodiscard]] const auto &get_components() const noexcept;

  [[nodiscard]] auto &get_components() noexcept;

  [[nodiscard]] auto get_size() const noexcept;

  void add_component() noexcept;

  void add_component(const GaussianComponent<Dim> &component) noexcept;

  void add_component(GaussianComponent<Dim> &&component) noexcept;

  void fit(const StaticRowsMatrix<Dim> &samples);

  [[nodiscard]] auto evaluate(gmix::ColVector<Dim> &sample) const;

  void reset() noexcept;

  using container_type = std::vector<GaussianComponent<Dim>>;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;

  [[nodiscard]] inline iterator begin() noexcept { return components_.begin(); }

  [[nodiscard]] inline const_iterator cbegin() const noexcept {
    return components_.cbegin();
  }

  [[nodiscard]] inline iterator end() noexcept { return components_.end(); }

  [[nodiscard]] inline const_iterator cend() const noexcept {
    return components_.cend();
  }

private:
  std::vector<GaussianComponent<Dim>> components_{};
};

template <int Dim, template <int> typename FittingStrategy>
GaussianMixture<Dim, FittingStrategy>::GaussianMixture() noexcept
    : FittingStrategy<Dim>() {}

template <int Dim, template <int> typename FittingStrategy>
GaussianMixture<Dim, FittingStrategy>::GaussianMixture(
    std::initializer_list<GaussianComponent<Dim>> components) noexcept
    : FittingStrategy<Dim>(), components_{components} {}

template <int Dim, template <int> typename FittingStrategy>
template <typename Parameters, typename>
GaussianMixture<Dim, FittingStrategy>::GaussianMixture(
    Parameters &&params) noexcept
    : FittingStrategy<Dim>(std::forward<Parameters>(params)) {}

template <int Dim, template <int> typename FittingStrategy>
template <typename Parameters, typename>
GaussianMixture<Dim, FittingStrategy>::GaussianMixture(
    std::initializer_list<GaussianComponent<Dim>> components,
    Parameters &&params) noexcept
    : FittingStrategy<Dim>(std::forward<Parameters>(params)),
      components_{components} {}

template <int Dim, template <int> typename FittingStrategy>
const auto &
GaussianMixture<Dim, FittingStrategy>::get_component(size_t idx) const {
  return components_.at(idx);
}

template <int Dim, template <int> typename FittingStrategy>
const auto &
GaussianMixture<Dim, FittingStrategy>::get_components() const noexcept {
  return components_;
}

template <int Dim, template <int> typename FittingStrategy>
auto &GaussianMixture<Dim, FittingStrategy>::get_components() noexcept {
  return components_;
}

template <int Dim, template <int> typename FittingStrategy>
auto GaussianMixture<Dim, FittingStrategy>::get_size() const noexcept {
  return components_.size();
}

template <int Dim, template <int> typename FittingStrategy>
void GaussianMixture<Dim, FittingStrategy>::add_component() noexcept {
  components_.emplace_back();
}

template <int Dim, template <int> typename FittingStrategy>
void GaussianMixture<Dim, FittingStrategy>::add_component(
    const GaussianComponent<Dim> &component) noexcept {
  components_.push_back(component);
};

template <int Dim, template <int> typename FittingStrategy>
void GaussianMixture<Dim, FittingStrategy>::add_component(
    GaussianComponent<Dim> &&component) noexcept {
  components_.emplace_back(std::move(component));
}

template <int Dim, template <int> typename FittingStrategy>
void GaussianMixture<Dim, FittingStrategy>::fit(
    const StaticRowsMatrix<Dim> &samples) {
  FittingStrategy<Dim>::fit(components_, samples);
}

template <int Dim, template <int> typename FittingStrategy>
auto GaussianMixture<Dim, FittingStrategy>::evaluate(
    gmix::ColVector<Dim> &sample) const {
  return std::accumulate(
      components_.begin(), components_.end(), 0.0,
      [&x = std::as_const(sample)](auto sum, const auto &component) -> double {
        return sum + component.evaluate(x);
      });
}

template <int Dim, template <int> typename FittingStrategy>
void GaussianMixture<Dim, FittingStrategy>::reset() noexcept {
  components_.resize(0);
}

template <int Dim, template <int> typename FittingStrategy>
std::ostream &operator<<(std::ostream &os,
                         const GaussianMixture<Dim, FittingStrategy> &gmm) {
  os << "GaussianMixture<" << Dim << ">\n";
  for (const auto &component : gmm.get_components())
    os << component << '\n';
  return os;
}

} // namespace gmix
#endif // !GMSAM_GAUSSIAN_MIXTURE_HPP
