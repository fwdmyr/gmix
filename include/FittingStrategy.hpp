#include "GaussianMixture.hpp"

namespace gm {

struct FittingStrategyParameters {};

template <int Dim> class FittingStrategy {
public:
  explicit FittingStrategy(const FittingStrategyParameters &parameters)
      : parameters_(parameters) {}
  virtual ~FittingStrategy();
  virtual void fit(GaussianMixture<Dim> &) const = 0;

protected:
  FittingStrategyParameters parameters_;
};

} // namespace gm
