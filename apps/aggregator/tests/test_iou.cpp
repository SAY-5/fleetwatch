#include <gtest/gtest.h>

#include "iou.h"

using fwagg::Box;
using fwagg::iou;

TEST(Iou, IdenticalBoxesIsOne) {
  Box a{0, 0, 10, 10};
  EXPECT_DOUBLE_EQ(iou(a, a), 1.0);
}

TEST(Iou, NoOverlapIsZero) {
  Box a{0, 0, 10, 10};
  Box b{20, 20, 30, 30};
  EXPECT_DOUBLE_EQ(iou(a, b), 0.0);
}

TEST(Iou, HalfOverlapHandComputed) {
  // a and b are 10x10; they overlap in a 5x10 strip -> inter=50.
  // union = 100 + 100 - 50 = 150. iou = 50/150 = 1/3.
  Box a{0, 0, 10, 10};
  Box b{5, 0, 15, 10};
  EXPECT_NEAR(iou(a, b), 1.0 / 3.0, 1e-12);
}

TEST(Iou, Symmetric) {
  Box a{1, 2, 9, 7};
  Box b{3, 1, 12, 8};
  EXPECT_DOUBLE_EQ(iou(a, b), iou(b, a));
}

TEST(Iou, TouchingEdgesIsZero) {
  Box a{0, 0, 10, 10};
  Box b{10, 0, 20, 10};
  EXPECT_DOUBLE_EQ(iou(a, b), 0.0);
}
