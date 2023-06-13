#ifndef GMSAM_GAUSSIAN_COMPONENT_HPP

#endif // !GMSAM_GAUSSIAN_COMPONENT_HPP
#include "Common.hpp"
#include <optional>

namespace gm {

namespace {

constexpr double GAUSSIAN_SCALER(int Dim) {
  return 1.0 / std::pow(2 * M_PI, 0.5 * Dim);
}

} // namespace

template <int Dim> class GaussianComponent {

public:
  GaussianComponent()
      : cache_(std::nullopt), weight_(0.0),
        mean_(static_cast<Vector<Dim>>(Vector<Dim>::Zero())),
        covariance_(static_cast<Matrix<Dim, Dim>>(Matrix<Dim, Dim>::Zero())),
        sqrt_information_(
            static_cast<Matrix<Dim, Dim>>(Matrix<Dim, Dim>::Zero())),
        llt_(){};
  GaussianComponent(double, const Vector<Dim> &, const Matrix<Dim, Dim> &);
  GaussianComponent(const GaussianComponent<Dim> &) = default;
  GaussianComponent(GaussianComponent<Dim> &&) = default;
  GaussianComponent<Dim> &operator=(const GaussianComponent<Dim> &) = default;
  GaussianComponent<Dim> &operator=(GaussianComponent<Dim> &&) = default;

  ~GaussianComponent() = default;

  double get_weight() const { return weight_; }
  const Vector<Dim> &get_mean() const { return mean_; }
  const Matrix<Dim, Dim> &get_covariance() const { return covariance_; }

  void set_weight(double weight) {
    if (weight_ == weight)
      return;
    weight_ = weight;
    cache_.reset();
  }
  void set_mean(const Vector<Dim> &mean) {
    if (mean_ == mean)
      return;
    mean_ = mean;
    cache_.reset();
  }
  void set_covariance(const Matrix<Dim, Dim> &covariance) {
    if (covariance_ == covariance)
      return;
    covariance_ = covariance;
    sqrt_information_ =
        Eigen::SelfAdjointEigenSolver<Matrix<Dim, Dim>>(covariance)
            .operatorInverseSqrt();
    llt_ = covariance.llt();
    cache_.reset();
  }

  double operator()(const Vector<Dim> &) const;

  friend std::ostream &operator<<(std::ostream &os,
                                  const GaussianComponent<Dim> &component) {
    os << "Component<" << Dim << ">" << '\n';
    os << "Weight:\n" << component.weight_ << '\n';
    os << "Mean:\n" << component.mean_ << '\n';
    os << "Covariance:\n" << component.covariance_ << '\n';
    return os;
  }

private:
  std::optional<double> cache_{};
  double weight_{};
  Vector<Dim> mean_{};
  Matrix<Dim, Dim> covariance_{};
  Matrix<Dim, Dim> sqrt_information_{};
  Eigen::LLT<Matrix<Dim, Dim>> llt_{};
};

template <int Dim>
GaussianComponent<Dim>::GaussianComponent(double weight,
                                          const Vector<Dim> &mean,
                                          const Matrix<Dim, Dim> &covariance)
    : cache_(std::nullopt), weight_(weight), mean_(mean),
      covariance_(covariance),
      sqrt_information_(
          Eigen::SelfAdjointEigenSolver<Matrix<Dim, Dim>>(covariance)
              .operatorInverseSqrt()),
      llt_(covariance.llt()) {}

template <int Dim>
double GaussianComponent<Dim>::operator()(const Vector<Dim> &x) const {
  if (!cache_)
    cache_ = GAUSSIAN_SCALER(Dim) * sqrt_information_.determinant();
  return *cache_ *
         std::exp(-0.5 * (llt_.matrixL().solve(x - mean_)).squaredNorm());
}

} // namespace gm
