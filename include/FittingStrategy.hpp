#ifndef GMSAM_FITTING_STRATEGY_HPP
#define GMSAM_FITTING_STRATEGY_HPP

#include "GaussianComponent.hpp"
#include "Statistics.hpp"
#include <vector>

namespace gm {

template <int Dim> class FittingStrategy {
public:
  virtual ~FittingStrategy();
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const std::vector<Vector<Dim>> &) const;
};

template <int Dim> class NoneStrategy final : public FittingStrategy<Dim> {
public:
  virtual ~NoneStrategy();
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const std::vector<Vector<Dim>> &) const override;
};

} // namespace gm
#endif // !GMSAM_FITTING_STRATEGY_HPP
