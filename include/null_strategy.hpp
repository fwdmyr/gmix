#ifndef GMSAM_NULL_STRATEGY_HPP
#define GMSAM_NULL_STRATEGY_HPP

#include "gaussian_component.hpp"
#include <stdexcept>
#include <vector>

namespace gmix {

struct NullStrategyUsedException : public std::runtime_error {
  NullStrategyUsedException() noexcept
      : std::runtime_error("Tried to use NullStrategy"){};
};

template <int Dim> class NullStrategy {
protected:
  NullStrategy() = default;

public:
  void fit(std::vector<GaussianComponent<Dim>> &,
           const StaticRowsMatrix<Dim> &) const {
    throw NullStrategyUsedException{};
  }

  void initialize(std::vector<GaussianComponent<Dim>> &,
                  const StaticRowsMatrix<Dim> &) const {
    throw NullStrategyUsedException{};
  }
};

} // namespace gmix
#endif // !GMSAM_NULL_STRATEGY_HPP
