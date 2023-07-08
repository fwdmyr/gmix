#ifndef GMSAM_TYPE_TRAITS_HPP
#define GMSAM_TYPE_TRAITS_HPP

#include "BaseStrategy.hpp"
#include "ExpectationMaximizationStrategy.hpp"
#include "KMeansStrategy.hpp"
#include "VariationalBayesianInferenceStrategy.hpp"

namespace gm {

namespace internal {

template <typename> struct StrategyTypeTraits;

template <int Dim> struct StrategyTypeTraits<gm::BaseStrategy<Dim>> {
  using ParameterType = typename gm::BaseStrategy<Dim>::Parameters;
  static constexpr int Dimension = Dim;
};

template <int Dim> struct StrategyTypeTraits<gm::KMeansStrategy<Dim>> {
  using ParameterType = typename gm::KMeansStrategy<Dim>::Parameters;
  static constexpr int Dimension = Dim;
};

template <int Dim>
struct StrategyTypeTraits<gm::ExpectationMaximizationStrategy<Dim>> {
  using ParameterType =
      typename gm::ExpectationMaximizationStrategy<Dim>::Parameters;
  static constexpr int Dimension = Dim;
};

template <int Dim>
struct StrategyTypeTraits<gm::VariationalBayesianInferenceStrategy<Dim>> {
  using ParameterType =
      typename gm::VariationalBayesianInferenceStrategy<Dim>::Parameters;
  static constexpr int Dimension = Dim;
};

} // namespace internal

} // namespace gm

#endif // !GMSAM_TYPE_TRAITS_HPP
