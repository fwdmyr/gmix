#include "TestHelpers.hpp"
namespace {

class GaussianComponentFixture : public testing::Test {
protected:
  static void SetUpTestSuite() {
    const auto weight = 0.5;
    const auto mean = (gm::ColVector<2>() << -2.3, 4.1).finished();
    const auto covariance =
        (gm::Matrix<2, 2>() << 3.3, 0.4, 0.4, 2.8).finished();
    component_ = {weight, mean, covariance};
    sample_ = (gm::ColVector<2>() << -2.1, 5.1).finished();
  }

  static gm::GaussianComponent<2> component_;
  static gm::ColVector<2> sample_;
};

gm::GaussianComponent<2> GaussianComponentFixture::component_{};
gm::ColVector<2> GaussianComponentFixture::sample_{};

TEST_F(
    GaussianComponentFixture,
    FunctionCallOperator_GivenInitialSetOfParameters_ExpectCorrectProbability) {
  const auto probability = component_(sample_);

  EXPECT_NEAR(probability, 0.02207882591462017, test::DETERMINISTIC_TOLERANCE);
}

TEST_F(GaussianComponentFixture,
       FunctionCallOperator_GivenNewSetOfParameters_ExpectCorrectProbability) {
  const auto new_weight = 0.2;
  const auto new_mean = (gm::ColVector<2>() << -2.3, 4.1).finished();
  const auto new_covariance =
      (gm::Matrix<2, 2>() << 3.3, 0.4, 0.4, 2.8).finished();
  const auto new_component =
      gm::GaussianComponent<2>{new_weight, new_mean, new_covariance};

  component_.set_weight(new_weight);
  component_.set_mean(new_mean);
  component_.set_covariance(new_covariance);

  EXPECT_TRUE(test::is_near(new_component(sample_), component_(sample_),
                            test::DETERMINISTIC_TOLERANCE));
}

} // namespace
