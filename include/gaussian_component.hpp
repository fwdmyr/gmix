#ifndef GMIX_GAUSSIAN_COMPONENT_HPP
#define GMIX_GAUSSIAN_COMPONENT_HPP

#include "common.hpp"
#include <gtest/gtest.h>
#include <optional>

namespace gmix {

namespace {

[[nodiscard]] static constexpr double GAUSSIAN_SCALER(int Dim) {
  return 1.0 / std::pow(2.0 * M_PI, 0.5 * Dim);
}

} // namespace

template <int Dim> class GaussianComponent {

public:
  GaussianComponent() = default;

  GaussianComponent(double weight, const ColVector<Dim> &mean,
                    const Matrix<Dim, Dim> &covariance) noexcept;

  [[nodiscard]] auto get_weight() const noexcept;

  [[nodiscard]] const auto &get_mean() const noexcept;

  [[nodiscard]] const auto &get_covariance() const noexcept;

  void set_weight(double weight) noexcept;

  void set_mean(const ColVector<Dim> &mean) noexcept;

  void set_covariance(const Matrix<Dim, Dim> &covariance);

  [[nodiscard]] auto evaluate(const ColVector<Dim> &x) const;

  void swap(GaussianComponent &other) noexcept;

private:
  mutable std::optional<double> cache_{std::nullopt};
  double weight_{0.0};
  ColVector<Dim> mean_{ColVector<Dim>::Zero()};
  Matrix<Dim, Dim> covariance_{Matrix<Dim, Dim>::Zero()};
  Eigen::LLT<Matrix<Dim, Dim>> llt_{};
};

template <int Dim>
GaussianComponent<Dim>::GaussianComponent(
    double weight, const ColVector<Dim> &mean,
    const Matrix<Dim, Dim> &covariance) noexcept
    : cache_(std::nullopt), weight_(weight), mean_(mean),
      covariance_(covariance), llt_(covariance.llt()) {}

template <int Dim> auto GaussianComponent<Dim>::get_weight() const noexcept {
  return weight_;
}

template <int Dim>
const auto &GaussianComponent<Dim>::get_mean() const noexcept {
  return mean_;
}

template <int Dim>
const auto &GaussianComponent<Dim>::get_covariance() const noexcept {
  return covariance_;
}

template <int Dim>
void GaussianComponent<Dim>::set_weight(double weight) noexcept {
  weight_ = weight;
}

template <int Dim>
void GaussianComponent<Dim>::set_mean(const ColVector<Dim> &mean) noexcept {
  mean_ = mean;
}

template <int Dim>
void GaussianComponent<Dim>::set_covariance(
    const Matrix<Dim, Dim> &covariance) {
  if (covariance_ == covariance)
    return;
  covariance_ = covariance;
  llt_ = covariance.llt();
  cache_.reset();
}

template <int Dim>
auto GaussianComponent<Dim>::evaluate(const ColVector<Dim> &x) const {
  if (!cache_) {
    cache_.emplace(GAUSSIAN_SCALER(Dim) / llt_.matrixL().determinant());
  }
  return weight_ * (*cache_) *
         std::exp(-0.5 * (llt_.matrixL().solve(x - mean_)).squaredNorm());
}

template <int Dim>
void GaussianComponent<Dim>::swap(GaussianComponent<Dim> &other) noexcept {
  std::swap(weight_, other.weight_);
  std::swap(mean_, other.mean_);
  std::swap(covariance_, other.covariance_);
}

template <int Dim>
std::ostream &operator<<(std::ostream &os,
                         const GaussianComponent<Dim> &component) {
  os << "Component<" << Dim << ">" << '\n';
  os << "Weight:\n" << component.get_weight() << '\n';
  os << "Mean:\n" << component.get_mean() << '\n';
  os << "Covariance:\n" << component.get_covariance() << '\n';
  return os;
}

} // namespace gmix

#endif // !GMIX_GAUSSIAN_COMPONENT_HPP
