#include "FittingStrategy.hpp"

namespace gm {

struct KMeansStrategyParameters : FittingStrategyParameters {
  int n_components;
};

template <int Dim> class KMeansStrategy final : public FittingStrategy<Dim> {
public:
  explicit KMeansStrategy(const KMeansStrategyParameters &parameters)
      : FittingStrategy<Dim>(parameters) {}
  virtual void fit(GaussianMixture<Dim> &,
                   const std::vector<Vector<Dim>> &) const override;

private:
  void initialize(GaussianMixture<Dim> &,
                  const std::vector<Vector<Dim>> &) const;
};

template <int Dim>
void KMeansStrategy<Dim>::fit(GaussianMixture<Dim> &gmm,
                              const std::vector<Vector<Dim>> &samples) const {
  // TODO: Continue here!
  initialize(gmm, samples);
}

template <int Dim>
void KMeansStrategy<Dim>::initialize(
    GaussianMixture<Dim> &gmm, const std::vector<Vector<Dim>> &samples) const {
  // TODO: Continue here!
  if (gmm.get_size() == this->parameters_.n_components)
    return;
  gmm.reset();
}

} // namespace gm
