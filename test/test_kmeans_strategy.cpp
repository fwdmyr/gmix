#include "../include/KMeansStrategy.hpp"
#include "TestHelpers.hpp"
#include <gtest/gtest.h>

namespace {

class PartitionSamplesFixture : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    samples_ = static_cast<gm::StaticRowsMatrix<2>>(
        (gm::Matrix<2, 4>() << 0.9, 1.0, 1.9, 2.0, 0.9, 1.0, 1.9, 2.0)
            .finished());
  }
  static gm::StaticRowsMatrix<2> samples_;
};

gm::StaticRowsMatrix<2> PartitionSamplesFixture::samples_{};

TEST_F(
    PartitionSamplesFixture,
    PartitionSamples_GivenSamplesAndZeroPartitionSize_ExpectEmptyPartitionVector) {
  const auto n_partitions = 0;

  const auto partitions =
      gm::internal::partition_samples(samples_, n_partitions);

  EXPECT_EQ(partitions.size(), 0);
}

TEST_F(
    PartitionSamplesFixture,
    PartitionSamples_GivenSamplesAndNonzeroPartitionSize_ExpectCorrectAssigmentToPartitions) {
  const auto n_partitions = 2;

  const auto partitions =
      gm::internal::partition_samples(samples_, n_partitions);

  EXPECT_EQ(partitions.size(), 2);
  EXPECT_EQ(partitions[0].cols(), 2);
  EXPECT_EQ(partitions[0](0, 0), 0.9);
  EXPECT_EQ(partitions[0](1, 0), 0.9);
  EXPECT_EQ(partitions[0](0, 1), 1.0);
  EXPECT_EQ(partitions[0](1, 1), 1.0);
  EXPECT_EQ(partitions[1].cols(), 2);
  EXPECT_EQ(partitions[1](0, 0), 1.9);
  EXPECT_EQ(partitions[1](1, 0), 1.9);
  EXPECT_EQ(partitions[1](0, 1), 2.0);
  EXPECT_EQ(partitions[1](1, 1), 2.0);
}

TEST(PartitionSamplesResponsibly, Dummy) {}

TEST(UpdateWeight, Dummy) {}

TEST(UpdateMean, Dummy) {}

TEST(UpdateCovariance, Dummy) {}

TEST(GetMeanMatrix, Dummy) {}

TEST(IsEarlyStoppingConditionFulfilled, Dummy) {}

class KMeansStrategyFixture : public ::testing::TestWithParam<bool> {
protected:
  static void SetUpTestSuite() {
    gmm_.add_component({0.5, (gm::Vector<2>() << 2.0, 9.0).finished(),
                        (gm::Matrix<2, 2>() << 2.0, 0.0, 0.0, 2.0).finished()});
    gmm_.add_component({0.5, (gm::Vector<2>() << -5.0, 4.0).finished(),
                        (gm::Matrix<2, 2>() << 0.5, 0.0, 0.0, 0.5).finished()});
    samples_ = gm::draw_from_gaussian_mixture(gmm_, 1E5);
  }

  void SetUp() override {
    gm::KMeansParameters<2> parameters;
    parameters.n_components = 2;
    parameters.n_iterations = 10;
    parameters.early_stopping_threshold = 0.0;
    parameters.warm_start = GetParam();
    strategy_ = gm::KMeansStrategy<2>{parameters};
  }

  static gm::GaussianMixture<2> gmm_;
  static gm::StaticRowsMatrix<2> samples_;
  gm::KMeansParameters<2> parameters_{};
  gm::KMeansStrategy<2> strategy_{gm::KMeansParameters<2>{}};
};

gm::GaussianMixture<2> KMeansStrategyFixture::gmm_{};
gm::StaticRowsMatrix<2> KMeansStrategyFixture::samples_{};

TEST_P(
    KMeansStrategyFixture,
    Fit_GivenParametersAndSamples_ExpectCorrectApproximationOfUnderlyingDistribution) {
  gm::GaussianMixture<2> gmm;
  if (GetParam()) {
    gmm.add_component({0.5, (gm::Vector<2>() << 1.0, 1.0).finished(),
                       (gm::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
    gmm.add_component({0.5, (gm::Vector<2>() << -1.0, -1.0).finished(),
                       (gm::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
  }

  gm::fit(samples_, strategy_, gmm);

  EXPECT_TRUE(
      test::compare_gaussian_mixtures(gmm_, gmm, test::RANDOM_TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(KMeansStrategyWarmColdStart, KMeansStrategyFixture,
                         testing::Values(true, false));

} // namespace
