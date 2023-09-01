#ifndef GMSAM_GAUSSIAN_COMPONENT_HPP
#define GMSAM_GAUSSIAN_COMPONENT_HPP

#include "common.hpp"
#include <optional>

namespace gmix {

namespace {

[[nodiscard]] static constexpr double GAUSSIAN_SCALER(int Dim) {
  return 1.0 / std::pow(2.0 * M_PI, 0.5 * Dim);
}

} // namespace

template <int Dim> class GaussianComponent {

public:
  GaussianComponent() noexcept
      : cache_(std::nullopt), weight_(0.0),
        mean_(static_cast<ColVector<Dim>>(ColVector<Dim>::Zero())),
        covariance_(static_cast<Matrix<Dim, Dim>>(Matrix<Dim, Dim>::Zero())),
        llt_(){};
  GaussianComponent(double, const ColVector<Dim> &,
                    const Matrix<Dim, Dim> &) noexcept;

  [[nodiscard]] double get_weight() const { return weight_; }
  [[nodiscard]] const ColVector<Dim> &get_mean() const { return mean_; }
  [[nodiscard]] const Matrix<Dim, Dim> &get_covariance() const {
    return covariance_;
  }

  void set_weight(double weight) { weight_ = weight; }
  void set_mean(const ColVector<Dim> &mean) { mean_ = mean; }
  void set_covariance(const Matrix<Dim, Dim> &covariance) {
    if (covariance_ == covariance)
      return;
    covariance_ = covariance;
    llt_ = covariance.llt();
    cache_.reset();
  }

  [[nodiscard]] double operator()(const ColVector<Dim> &) const;

  friend std::ostream &operator<<(std::ostream &os,
                                  const GaussianComponent<Dim> &component) {
    os << "Component<" << Dim << ">" << '\n';
    os << "Weight:\n" << component.weight_ << '\n';
    os << "Mean:\n" << component.mean_ << '\n';
    os << "Covariance:\n" << component.covariance_ << '\n';
    return os;
  }

private:
  mutable std::optional<double> cache_{};
  double weight_{};
  ColVector<Dim> mean_{};
  Matrix<Dim, Dim> covariance_{};
  Eigen::LLT<Matrix<Dim, Dim>> llt_{};
};

template <int Dim>
GaussianComponent<Dim>::GaussianComponent(
    double weight, const ColVector<Dim> &mean,
    const Matrix<Dim, Dim> &covariance) noexcept
    : cache_(std::nullopt), weight_(weight), mean_(mean),
      covariance_(covariance), llt_(covariance.llt()) {}

template <int Dim>
double GaussianComponent<Dim>::operator()(const ColVector<Dim> &x) const {
  if (!cache_) {
    cache_.emplace(GAUSSIAN_SCALER(Dim) / llt_.matrixL().determinant());
  }
  return weight_ * (*cache_) *
         std::exp(-0.5 * (llt_.matrixL().solve(x - mean_)).squaredNorm());
}

} // namespace gmix

#endif // !GMSAM_GAUSSIAN_COMPONENT_HPP
