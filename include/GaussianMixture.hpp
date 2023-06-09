#include "GaussianComponent.hpp"
#include <initializer_list>
#include <vector>

namespace gm {

template <int Dim> class GaussianMixture {

public:
  using container_type = std::vector<GaussianComponent<Dim>>;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;

  GaussianMixture() = default;
  GaussianMixture(std::initializer_list<GaussianComponent<Dim>>);
  GaussianMixture(double, const Vector<Dim> &, const Matrix<Dim, Dim> &);
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

private:
  std::vector<GaussianComponent<Dim>> components_;
};

} // namespace gm
