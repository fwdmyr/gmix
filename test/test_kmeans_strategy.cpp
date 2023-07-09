#include "../include/KMeansStrategy.hpp"
#include "TestHelpers.hpp"
#include <gtest/gtest.h>

namespace {

class PartitionSamplesFixture : public testing::Test {
protected:
  void SetUp() override {
    samples_ = (gm::Matrix<2U, 4U>() << 0.9, 1.0, 1.9, 2.0, 0.9, 1.0, 1.9, 2.0)
                   .finished();
  }
  gm::StaticRowsMatrix<2U> samples_;
};

TEST_F(PartitionSamplesFixture,
       GivenSamplesAndZeroPartitionSize_ExpectEmptyPartitionVector) {
  const auto n_partitions = 0;

  const auto partitions =
      gm::internal::partition_samples(samples_, n_partitions);

  EXPECT_EQ(partitions.size(), 0);
}

TEST_F(PartitionSamplesFixture,
       GivenSamplesAndNonzeroPartitionSize_ExpectCorrectAssigmentToPartitions) {
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

} // namespace
