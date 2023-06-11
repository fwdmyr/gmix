#include "FittingStrategy.hpp"

namespace gm {

struct KMeansStrategyParameters : FittingStrategyParameters {};

template <int Dim> class KMeansStrategy final : public FittingStrategy<Dim> {
public:
  explicit KMeansStrategy(const KMeansStrategyParameters &parameters)
      : FittingStrategy<Dim>(parameters) {}
  virtual void fit(GaussianMixture<Dim> &) const override;
};

} // namespace gm
