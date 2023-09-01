#ifndef GMSAM_BASE_STRATEGY_HPP
#define GMSAM_BASE_STRATEGY_HPP

#include "gaussian_component.hpp"
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

} // namespace gmix
#endif // !GMSAM_BASE_STRATEGY_HPP
