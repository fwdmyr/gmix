#ifndef GMSAM_BASE_STRATEGY_HPP
#define GMSAM_BASE_STRATEGY_HPP

#include "gaussian_component.hpp"
#include "matrix_traits.hpp"
#include "statistics.hpp"
#include <stdexcept>
#include <vector>

namespace gmix {

template <int Dim> class BaseStrategy {
public:
  virtual ~BaseStrategy() = default;

  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const StaticRowsMatrix<Dim> &) const = 0;

  virtual void initialize(std::vector<GaussianComponent<Dim>> &,
                          const StaticRowsMatrix<Dim> &) const = 0;
};

template <int Dim> class NullStrategy : public BaseStrategy<Dim> {
protected:
  NullStrategy() = default;

public:
  using ParamType = void;

  void fit(std::vector<GaussianComponent<Dim>> &,
           const StaticRowsMatrix<Dim> &) const override {}

  void initialize(std::vector<GaussianComponent<Dim>> &,
                  const StaticRowsMatrix<Dim> &) const override {}
};

} // namespace gmix
#endif // !GMSAM_BASE_STRATEGY_HPP
