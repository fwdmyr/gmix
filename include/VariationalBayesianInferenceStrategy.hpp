#ifndef GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_HPP
#define GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_HPP

#include "BaseStrategy.hpp"
#include "Common.hpp"
#include "Statistics.hpp"
#include <limits>
#include <random>

namespace gm {

template <int Dim>
class VariationalBayesianInferenceStrategy final : public BaseStrategy<Dim> {
public:
  struct Parameters {
    int n_components;
    int n_iterations;
  };

  explicit VariationalBayesianInferenceStrategy(const Parameters &parameters)
      : parameters_(parameters) {}
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const StaticRowsMatrix<Dim> &) const override;

private:
  Parameters parameters_;
};

} // namespace gm

#endif // !GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_HPP
