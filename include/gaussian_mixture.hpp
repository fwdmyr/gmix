#ifndef GMIX_GAUSSIAN_MIXTURE_HPP
#define GMIX_GAUSSIAN_MIXTURE_HPP

#include "gaussian_component.hpp"
#include "null_policy.hpp"
#include "statistics.hpp"
#include <memory>
#include <random>
#include <vector>

namespace gmix {

template <int Dim, template <int> typename FittingPolicy = NullPolicy>
class GaussianMixture : public FittingPolicy<Dim> {

public:
  explicit GaussianMixture() noexcept;

  explicit GaussianMixture(
      std::initializer_list<GaussianComponent<Dim>>) noexcept;

  template <typename Parameters,
            typename = std::enable_if_t<is_constructible_with_v<
                FittingPolicy<Dim>, std::decay_t<Parameters>>>>
  explicit GaussianMixture(Parameters &&) noexcept;

  template <typename Parameters,
            typename = std::enable_if_t<is_constructible_with_v<
                FittingPolicy<Dim>, std::decay_t<Parameters>>>>
  GaussianMixture(std::initializer_list<GaussianComponent<Dim>>,
                  Parameters &&) noexcept;

  template <template <int> typename OtherFittingPolicy>
  GaussianMixture &
  operator=(const GaussianMixture<Dim, OtherFittingPolicy> &other);

  template <template <int> typename OtherFittingPolicy>
  GaussianMixture &operator=(GaussianMixture<Dim, OtherFittingPolicy> &&other);

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

  template <template <int> typename OtherFittingPolicy>
  void swap(GaussianMixture<Dim, OtherFittingPolicy> &other) noexcept;

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

template <int Dim, template <int> typename FittingPolicy>
GaussianMixture<Dim, FittingPolicy>::GaussianMixture() noexcept
    : FittingPolicy<Dim>() {}

template <int Dim, template <int> typename FittingPolicy>
GaussianMixture<Dim, FittingPolicy>::GaussianMixture(
    std::initializer_list<GaussianComponent<Dim>> components) noexcept
    : FittingPolicy<Dim>(), components_{components} {}

template <int Dim, template <int> typename FittingPolicy>
template <typename Parameters, typename>
GaussianMixture<Dim, FittingPolicy>::GaussianMixture(
    Parameters &&params) noexcept
    : FittingPolicy<Dim>(std::forward<Parameters>(params)) {}

template <int Dim, template <int> typename FittingPolicy>
template <typename Parameters, typename>
GaussianMixture<Dim, FittingPolicy>::GaussianMixture(
    std::initializer_list<GaussianComponent<Dim>> components,
    Parameters &&params) noexcept
    : FittingPolicy<Dim>(std::forward<Parameters>(params)),
      components_{components} {}

template <int Dim, template <int> typename FittingPolicy>
template <template <int> typename OtherFittingPolicy>
GaussianMixture<Dim, FittingPolicy> &
GaussianMixture<Dim, FittingPolicy>::operator=(
    const GaussianMixture<Dim, OtherFittingPolicy> &other) {
  auto tmp = GaussianMixture<Dim, OtherFittingPolicy>{other};
  swap(tmp);
  return *this;
}

template <int Dim, template <int> typename FittingPolicy>
template <template <int> typename OtherFittingPolicy>
GaussianMixture<Dim, FittingPolicy> &
GaussianMixture<Dim, FittingPolicy>::operator=(
    GaussianMixture<Dim, OtherFittingPolicy> &&other) {
  swap(other);
  return *this;
}

template <int Dim, template <int> typename FittingPolicy>
const auto &
GaussianMixture<Dim, FittingPolicy>::get_component(size_t idx) const {
  return components_.at(idx);
}

template <int Dim, template <int> typename FittingPolicy>
const auto &
GaussianMixture<Dim, FittingPolicy>::get_components() const noexcept {
  return components_;
}

template <int Dim, template <int> typename FittingPolicy>
auto &GaussianMixture<Dim, FittingPolicy>::get_components() noexcept {
  return components_;
}

template <int Dim, template <int> typename FittingPolicy>
auto GaussianMixture<Dim, FittingPolicy>::get_size() const noexcept {
  return components_.size();
}

template <int Dim, template <int> typename FittingPolicy>
void GaussianMixture<Dim, FittingPolicy>::add_component() noexcept {
  components_.emplace_back();
}

template <int Dim, template <int> typename FittingPolicy>
void GaussianMixture<Dim, FittingPolicy>::add_component(
    const GaussianComponent<Dim> &component) noexcept {
  components_.push_back(component);
};

template <int Dim, template <int> typename FittingPolicy>
void GaussianMixture<Dim, FittingPolicy>::add_component(
    GaussianComponent<Dim> &&component) noexcept {
  components_.emplace_back(std::move(component));
}

template <int Dim, template <int> typename FittingPolicy>
void GaussianMixture<Dim, FittingPolicy>::fit(
    const StaticRowsMatrix<Dim> &samples) {
  FittingPolicy<Dim>::fit(components_, samples);
}

template <int Dim, template <int> typename FittingPolicy>
auto GaussianMixture<Dim, FittingPolicy>::evaluate(
    gmix::ColVector<Dim> &sample) const {
  return std::accumulate(
      components_.begin(), components_.end(), 0.0,
      [&x = std::as_const(sample)](auto sum, const auto &component) -> double {
        return sum + component.evaluate(x);
      });
}

template <int Dim, template <int> typename FittingPolicy>
void GaussianMixture<Dim, FittingPolicy>::reset() noexcept {
  components_.resize(0);
}

template <int Dim, template <int> typename FittingPolicy>
template <template <int> typename OtherFittingPolicy>
void GaussianMixture<Dim, FittingPolicy>::swap(
    GaussianMixture<Dim, OtherFittingPolicy> &other) noexcept {
  std::swap(components_, other.components_);
}

template <int Dim, template <int> typename FittingPolicy>
std::ostream &operator<<(std::ostream &os,
                         const GaussianMixture<Dim, FittingPolicy> &gmm) {
  os << "GaussianMixture<" << Dim << ">\n";
  for (const auto &component : gmm.get_components())
    os << component << '\n';
  return os;
}

} // namespace gmix
#endif // !GMIX_GAUSSIAN_MIXTURE_HPP
