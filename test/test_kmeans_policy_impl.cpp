#include "../include/kmeans_policy_impl.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>

namespace {

class PartitionSamplesFixture : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    samples_ = (gmix::StaticRowsMatrix<2>(2, 4) << 0.9, 1.0, 1.9, 2.0, 0.9, 1.0,
                1.9, 2.0)
                   .finished();
  }

  static gmix::StaticRowsMatrix<2> samples_;
};

gmix::StaticRowsMatrix<2> PartitionSamplesFixture::samples_{};

TEST_F(
    PartitionSamplesFixture,
    PartitionSamples_GivenSamplesAndZeroPartitionSize_ExpectEmptyPartitionColVector) {
  const auto n_partitions = 0;

  const auto partitions =
      gmix::internal::partition_samples(samples_, n_partitions);

  EXPECT_EQ(partitions.size(), 0);
}

TEST_F(
    PartitionSamplesFixture,
    PartitionSamples_GivenSamplesAndNonzeroPartitionSize_ExpectCorrectAssigmentToPartitions) {
  const auto n_partitions = 2;

  const auto partitions =
      gmix::internal::partition_samples(samples_, n_partitions);

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

TEST(PartitionSamplesResponsibly,
     GivenSamplesAndGaussianMixture_ExpectCorrectAssociation) {
  auto gmm = gmix::GaussianMixture<2>{};
  gmm.add_component({0.5, (gmix::ColVector<2>() << 1.0, 1.0).finished(),
                     (gmix::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gmm.add_component({0.5, (gmix::ColVector<2>() << 2.0, 2.0).finished(),
                     (gmix::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
  const auto samples = (gmix::StaticRowsMatrix<2>(2, 4) << 0.9, 1.0, 1.9, 2.0,
                        0.9, 1.0, 1.9, 2.0)
                           .finished();

  const auto partitions = gmix::internal::partition_samples_responsibly(
      gmm.get_components(), samples);

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
};

TEST(UpdateWeight, GivenPartitionedSamples_ExpectCorrectWeightUpdate) {
  auto gmm = gmix::GaussianMixture<2>{};
  gmm.add_component();
  gmm.add_component();
  auto partitions = std::vector<gmix::StaticRowsMatrix<2>>{};
  partitions.push_back(
      (gmix::StaticRowsMatrix<2>(2, 2) << 0.9, 1.0, 0.9, 1.0).finished());
  partitions.push_back(
      (gmix::StaticRowsMatrix<2>(2, 2) << 1.9, 2.0, 1.9, 2.0).finished());

  gmix::internal::update_weight(gmm.get_components(), partitions);

  EXPECT_EQ(gmm.get_component(0).get_weight(), 0.5);
  EXPECT_EQ(gmm.get_component(1).get_weight(), 0.5);
}

TEST(UpdateMean, GivenPartitionedSamples_ExpectCorrectMeanUpdate) {
  auto gmm = gmix::GaussianMixture<2>{};
  gmm.add_component();
  gmm.add_component();
  auto partitions = std::vector<gmix::StaticRowsMatrix<2>>{};
  partitions.push_back(
      (gmix::StaticRowsMatrix<2>(2, 2) << 0.9, 1.0, 0.9, 1.0).finished());
  partitions.push_back(
      (gmix::StaticRowsMatrix<2>(2, 2) << 1.9, 2.0, 1.9, 2.0).finished());

  gmix::internal::update_mean(gmm.get_components(), partitions);

  EXPECT_EQ(gmm.get_component(0).get_mean()(0, 0), 0.95);
  EXPECT_EQ(gmm.get_component(0).get_mean()(1, 0), 0.95);
  EXPECT_EQ(gmm.get_component(1).get_mean()(0, 0), 1.95);
  EXPECT_EQ(gmm.get_component(1).get_mean()(1, 0), 1.95);
}

TEST(UpdateCovariance, GivenPartitionedSamples_ExpectCorrectCovarianceUpdate) {
  auto gmm = gmix::GaussianMixture<2>{};
  gmm.add_component();
  gmm.add_component();
  auto partitions = std::vector<gmix::StaticRowsMatrix<2>>{};
  partitions.push_back(
      (gmix::StaticRowsMatrix<2>(2, 2) << 0.9, 1.0, 0.9, 1.0).finished());
  partitions.push_back(
      (gmix::StaticRowsMatrix<2>(2, 2) << 1.9, 2.0, 1.9, 2.0).finished());
  const auto expected_covariance =
      (gmix::Matrix<2, 2>() << 0.005, 0.005, 0.005, 0.005).finished();

  gmix::internal::update_covariance(gmm.get_components(), partitions);

  EXPECT_TRUE(test::is_near(gmm.get_component(0).get_covariance(),
                            expected_covariance,
                            test::DETERMINISTIC_TOLERANCE));
  EXPECT_TRUE(test::is_near(gmm.get_component(1).get_covariance(),
                            expected_covariance,
                            test::DETERMINISTIC_TOLERANCE));
}

TEST(GetMeanMatrix, GivenGaussianMixture_ExpectCorrectMeanMatrix) {
  auto gmm = gmix::GaussianMixture<2>{};
  gmm.add_component({0.5, (gmix::ColVector<2>() << -1.1, 3.2).finished(),
                     (gmix::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});
  gmm.add_component({0.5, (gmix::ColVector<2>() << 4.6, -2.0).finished(),
                     (gmix::Matrix<2, 2>() << 1.0, 0.0, 0.0, 1.0).finished()});

  const auto mean_matrix =
      gmix::internal::get_mean_matrix(gmm.get_components());

  EXPECT_EQ(mean_matrix(0, 0), -1.1);
  EXPECT_EQ(mean_matrix(1, 0), 3.2);
  EXPECT_EQ(mean_matrix(0, 1), 4.6);
  EXPECT_EQ(mean_matrix(1, 1), -2.0);
}

TEST(IsEarlyStoppingConditionFulfilled,
     GivenMeanMatricesAndSmallEnoughThreshold_ExpectFalse) {
  const auto mean_matrix =
      (gmix::StaticRowsMatrix<2>(2, 2) << 0.9, 1.9, 0.99, 1.99).finished();
  const auto other_mean_matrix =
      (gmix::StaticRowsMatrix<2>(2, 2) << 1.0, 2.0, 1.0, 2.0).finished();
  const auto threshold = 0.1;

  const auto flag = gmix::internal::is_early_stopping_condition_fulfilled(
      mean_matrix, other_mean_matrix, threshold);

  EXPECT_FALSE(flag);
}

TEST(IsEarlyStoppingConditionFulfilled,
     GivenMeanMatricesAndLargeEnoughThreshold_ExpectTrue) {
  const auto mean_matrix =
      (gmix::StaticRowsMatrix<2>(2, 2) << 0.9, 1.9, 0.99, 1.99).finished();
  const auto other_mean_matrix =
      (gmix::StaticRowsMatrix<2>(2, 2) << 1.0, 2.0, 1.0, 2.0).finished();
  const auto threshold = 0.2;

  const auto flag = gmix::internal::is_early_stopping_condition_fulfilled(
      mean_matrix, other_mean_matrix, threshold);

  EXPECT_TRUE(flag);
}

} // namespace
