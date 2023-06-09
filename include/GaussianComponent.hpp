#include "Common.hpp"

namespace gm {

template <int Dim> class GaussianComponent {

public:
  GaussianComponent() = delete;
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
