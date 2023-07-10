#include "../include/Common.hpp"
#include "TestHelpers.hpp"
#include <gtest/gtest.h>

namespace {

class InitializeFixture : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    matrix_ = static_cast<gmix::Matrix<2, 2>>(
        (gmix::Matrix<2, 2>() << 1, 2, 3, 4).finished());
    vector_ = static_cast<gmix::ColVector<4>>(
        (gmix::ColVector<4>() << 1, 2, 3, 4).finished());
  }

  static gmix::Matrix<2, 2> matrix_;
  static gmix::ColVector<4> vector_;
};

gmix::Matrix<2, 2> InitializeFixture::matrix_{};
gmix::ColVector<4> InitializeFixture::vector_{};

TEST_F(
    InitializeFixture,
    GivenInitializerListOfInitializerLists_ExpectCorrectStaticRowsStaticColsMatrixInitialization) {
  const auto matrix = gmix::initialize<gmix::Matrix<2, 2>>({{1, 2}, {3, 4}});

  EXPECT_EQ(matrix.rows(), matrix_.rows());
  EXPECT_EQ(matrix.cols(), matrix_.cols());
  EXPECT_EQ(matrix(0, 0), matrix_(0, 0));
  EXPECT_EQ(matrix(1, 0), matrix_(1, 0));
  EXPECT_EQ(matrix(0, 1), matrix_(0, 1));
  EXPECT_EQ(matrix(1, 1), matrix_(1, 1));
}

TEST_F(
    InitializeFixture,
    GivenInitializerListOfInitializerLists_ExpectCorrectStaticRowsDynamicColsMatrixInitialization) {
  const auto matrix =
      gmix::initialize<gmix::StaticRowsMatrix<2>>({{1, 2}, {3, 4}});

  EXPECT_EQ(matrix.rows(), matrix_.rows());
  EXPECT_EQ(matrix.cols(), matrix_.cols());
  EXPECT_EQ(matrix(0, 0), matrix_(0, 0));
  EXPECT_EQ(matrix(1, 0), matrix_(1, 0));
  EXPECT_EQ(matrix(0, 1), matrix_(0, 1));
  EXPECT_EQ(matrix(1, 1), matrix_(1, 1));
}

TEST_F(
    InitializeFixture,
    GivenInitializerListOfInitializerLists_ExpectCorrectDynamicRowsStaticColsMatrixInitialization) {
  const auto matrix =
      gmix::initialize<gmix::StaticColsMatrix<2>>({{1, 2}, {3, 4}});

  EXPECT_EQ(matrix.rows(), matrix_.rows());
  EXPECT_EQ(matrix.cols(), matrix_.cols());
  EXPECT_EQ(matrix(0, 0), matrix_(0, 0));
  EXPECT_EQ(matrix(1, 0), matrix_(1, 0));
  EXPECT_EQ(matrix(0, 1), matrix_(0, 1));
  EXPECT_EQ(matrix(1, 1), matrix_(1, 1));
}

TEST_F(
    InitializeFixture,
    GivenInitializerListOfInitializerLists_ExpectCorrectDynamicRowsDynamicColsMatrixInitialization) {
  const auto matrix = gmix::initialize<gmix::MatrixX>({{1, 2}, {3, 4}});

  EXPECT_EQ(matrix.rows(), matrix_.rows());
  EXPECT_EQ(matrix.cols(), matrix_.cols());
  EXPECT_EQ(matrix(0, 0), matrix_(0, 0));
  EXPECT_EQ(matrix(1, 0), matrix_(1, 0));
  EXPECT_EQ(matrix(0, 1), matrix_(0, 1));
  EXPECT_EQ(matrix(1, 1), matrix_(1, 1));
}

TEST_F(
    InitializeFixture,
    GivenInitializerList_ExpectCorrectColVectorWithStaticRowsInitialization) {
  const auto vector = gmix::initialize<gmix::ColVector<4>>({1, 2, 3, 4});

  EXPECT_EQ(vector.rows(), vector_.rows());
  EXPECT_EQ(vector.cols(), vector_.cols());
  EXPECT_EQ(vector(0, 0), vector_(0, 0));
  EXPECT_EQ(vector(1, 0), vector_(1, 0));
  EXPECT_EQ(vector(2, 0), vector_(2, 0));
  EXPECT_EQ(vector(3, 0), vector_(3, 0));
}

TEST_F(
    InitializeFixture,
    GivenInitializerList_ExpectCorrectColVectorWithDynamicRowsInitialization) {
  const auto vector = gmix::initialize<gmix::ColVectorX>({1, 2, 3, 4});

  EXPECT_EQ(vector.rows(), vector_.rows());
  EXPECT_EQ(vector.cols(), vector_.cols());
  EXPECT_EQ(vector(0, 0), vector_(0, 0));
  EXPECT_EQ(vector(1, 0), vector_(1, 0));
  EXPECT_EQ(vector(2, 0), vector_(2, 0));
  EXPECT_EQ(vector(3, 0), vector_(3, 0));
}

TEST_F(
    InitializeFixture,
    GivenInitializerList_ExpectCorrectRowVectorWithStaticColsInitialization) {
  const auto vector = gmix::initialize<gmix::RowVector<4>>({1, 2, 3, 4});

  std::cerr << vector << std::endl;

  EXPECT_EQ(vector.cols(), vector_.rows());
  EXPECT_EQ(vector.rows(), vector_.cols());
  EXPECT_EQ(vector(0, 0), vector_(0, 0));
  EXPECT_EQ(vector(0, 1), vector_(1, 0));
  EXPECT_EQ(vector(0, 2), vector_(2, 0));
  EXPECT_EQ(vector(0, 3), vector_(3, 0));
}

TEST_F(
    InitializeFixture,
    GivenInitializerList_ExpectCorrectRowVectorWithDynamicColsInitialization) {
  const auto vector = gmix::initialize<gmix::RowVectorX>({1, 2, 3, 4});

  std::cerr << vector << std::endl;

  EXPECT_EQ(vector.cols(), vector_.rows());
  EXPECT_EQ(vector.rows(), vector_.cols());
  EXPECT_EQ(vector(0, 0), vector_(0, 0));
  EXPECT_EQ(vector(0, 1), vector_(1, 0));
  EXPECT_EQ(vector(0, 2), vector_(2, 0));
  EXPECT_EQ(vector(0, 3), vector_(3, 0));
}

} // namespace
