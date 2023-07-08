#include "TestHelpers.hpp"
namespace {

class GaussianComponentTestFixture : public testing::Test {
protected:
  void SetUp() override {
    weight_ = 0.5;
    mean_ = (gm::Vector<2U>() << -2.3, 4.1).finished();
    covariance_ = (gm::Matrix<2U, 2U>() << 3.3, 0.4, 0.4, 2.8).finished();
    component_ = {weight_, mean_, covariance_};
    sample_ = (gm::Vector<2U>() << -2.1, 5.1).finished();
  }

  double weight_{};
  gm::Vector<2U> mean_{};
  gm::Matrix<2U, 2U> covariance_{};
  gm::GaussianComponent<2U> component_{};
  gm::Vector<2U> sample_{};
};
} // namespace

TEST_F(GaussianComponentTestFixture,
       TestFunctionCallOperatorWithInitialParameters) {
  const auto probability = component_(sample_);

  EXPECT_NEAR(probability, 0.02207882591462017, test::DETERMINISTIC_TOLERANCE);
}

TEST_F(GaussianComponentTestFixture,
       TestFunctionCallOperatorWithNewParameters) {
  const auto new_weight = 0.2;
  const auto new_mean = (gm::Vector<2U>() << -2.3, 4.1).finished();
  const auto new_covariance =
      (gm::Matrix<2U, 2U>() << 3.3, 0.4, 0.4, 2.8).finished();
  const auto new_component =
      gm::GaussianComponent<2U>{new_weight, new_mean, new_covariance};

  component_.set_weight(new_weight);
  component_.set_mean(new_mean);
  component_.set_covariance(new_covariance);

  EXPECT_TRUE(test::is_near(new_component(sample_), component_(sample_),
                            test::DETERMINISTIC_TOLERANCE));
}
