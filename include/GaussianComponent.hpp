#ifndef GMSAM_GAUSSIAN_COMPONENT_HPP

#endif // !GMSAM_GAUSSIAN_COMPONENT_HPP
#include "Common.hpp"

namespace gm {

template <int Dim> class GaussianComponent {

public:
  GaussianComponent()
      : weight_(0.0), mean_(static_cast<Vector<Dim>>(Vector<Dim>::Zero())),
        covariance_(static_cast<Matrix<Dim, Dim>>(Matrix<Dim, Dim>::Zero())){};
  GaussianComponent(double, const Vector<Dim> &, const Matrix<Dim, Dim> &);
  GaussianComponent(const GaussianComponent<Dim> &) = default;
  GaussianComponent(GaussianComponent<Dim> &&) = default;
  GaussianComponent<Dim> &operator=(const GaussianComponent<Dim> &) = default;
  GaussianComponent<Dim> &operator=(GaussianComponent<Dim> &&) = default;

  ~GaussianComponent() = default;

  double get_weight() const { return weight_; }
  const Vector<Dim> &get_mean() const { return mean_; }
  const Matrix<Dim, Dim> &get_covariance() const { return covariance_; }

  void set_weight(double weight) { weight_ = weight; }
  void set_mean(const Vector<Dim> &mean) { mean_ = mean; }
  void set_covariance(const Matrix<Dim, Dim> &covariance) {
    covariance_ = covariance;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const GaussianComponent<Dim> &component) {
    os << "Component<" << Dim << ">" << '\n';
    os << "Weight:\n" << component.weight_ << '\n';
    os << "Mean:\n" << component.mean_ << '\n';
    os << "Covariance:\n" << component.covariance_ << '\n';
    return os;
  }

private:
  double weight_{};
  Vector<Dim> mean_{};
  Matrix<Dim, Dim> covariance_{};
};

template <int Dim>
GaussianComponent<Dim>::GaussianComponent(double weight,
                                          const Vector<Dim> &mean,
                                          const Matrix<Dim, Dim> &covariance)
    : weight_(weight), mean_(mean), covariance_(covariance) {}

} // namespace gm
