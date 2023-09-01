#include "test_helpers.hpp"
namespace {

class GaussianComponentFixture : public testing::Test {
protected:
  static void SetUpTestSuite() {
    const auto weight = 0.5;
    const auto mean = (gmix::ColVector<2>() << -2.3, 4.1).finished();
    const auto covariance =
        (gmix::Matrix<2, 2>() << 3.3, 0.4, 0.4, 2.8).finished();
    component_ = {weight, mean, covariance};
    sample_ = (gmix::ColVector<2>() << -2.1, 5.1).finished();
  }

  static gmix::GaussianComponent<2> component_;
  static gmix::ColVector<2> sample_;
};

gmix::GaussianComponent<2> GaussianComponentFixture::component_{};
gmix::ColVector<2> GaussianComponentFixture::sample_{};

TEST_F(
    GaussianComponentFixture,
    FunctionCallOperator_GivenInitialSetOfParameters_ExpectCorrectProbability) {
  const auto probability = component_(sample_);

  EXPECT_NEAR(probability, 0.02207882591462017, test::DETERMINISTIC_TOLERANCE);
}

TEST_F(GaussianComponentFixture,
       FunctionCallOperator_GivenNewSetOfParameters_ExpectCorrectProbability) {
  const auto new_weight = 0.5;
  const auto new_mean = (gmix::ColVector<2>() << -2.3, 4.1).finished();
  const auto new_covariance =
      (gmix::Matrix<2, 2>() << 3.3, 0.4, 0.4, 2.8).finished();
  const auto new_component =
      gmix::GaussianComponent<2>{new_weight, new_mean, new_covariance};

  EXPECT_TRUE(test::is_near(new_component(sample_), component_(sample_),
                            test::DETERMINISTIC_TOLERANCE));
}

} // namespace
