#ifndef GMSAM_BASE_STRATEGY_HPP
#define GMSAM_BASE_STRATEGY_HPP

#include "GaussianComponent.hpp"
#include "Statistics.hpp"
#include "TypeTraits.hpp"
#include <stdexcept>
#include <vector>

namespace gmix {

template <int Dim> class BaseStrategy {
public:
  struct Parameters;

  virtual ~BaseStrategy() = default;
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const StaticRowsMatrix<Dim> &) const = 0;

protected:
  virtual void initialize(std::vector<GaussianComponent<Dim>> &,
                          const StaticRowsMatrix<Dim> &) const = 0;
};

} // namespace gmix
#endif // !GMSAM_BASE_STRATEGY_HPP
