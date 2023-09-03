# gmix

A library that provides a Gaussian mixture model and various efficient parameter fitting policies.
The current policy set consists of:
- k-Means clustering with covariance estimation
- Expectation-Maximization
- Variational Bayesian inference

Usage example:

```cpp
// Create a Gaussian mixture without policy.
auto fitted_gmm = gmix::GaussianMixture<Dim>{};

... // Add components to the Gaussian mixture

// Generate or load sample data.
const auto samples = gmix::draw_from_gaussian_mixture(fitted_gmm, n_samples);

// Parameterize the fitting policy.
auto parameters = gmix::VariationalBayesianInferenceParameters<Dim>{};

... // Adjust the parameters

// Fit a Gaussian mixture model with policy to the samples.
auto gmm = gmix::GaussianMixture<Dim, gmix::VariationalBayesianInferencePolicy>{std::move(parameters)};
gmm.fit(samples);
```
