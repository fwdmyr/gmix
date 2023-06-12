#ifndef GMSAM_BASE_STRATEGY_HPP
#define GMSAM_BASE_STRATEGY_HPP

#include "GaussianComponent.hpp"
#include "Statistics.hpp"
#include <stdexcept>
#include <vector>

namespace gm {

template <int Dim> class BaseStrategy {
public:
  virtual ~BaseStrategy() = default;
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const std::vector<Vector<Dim>> &) const {
    throw std::runtime_error("BaseStrategy::fit");
  };
};

} // namespace gm
#endif // !GMSAM_BASE_STRATEGY_HPP
