#include <gtest/gtest.h>

#include "map.h"
#include "match.h"

using fwagg::aggregate;
using fwagg::Det;
using fwagg::Gt;
using fwagg::match;

TEST(Aggregate, PerfectFrameMapIsOne) {
  std::vector<Det> dets{{"car", {0, 0, 10, 10}, 0.9}, {"person", {20, 20, 30, 30}, 0.8}};
  std::vector<Gt> gts{{"car", {0, 0, 10, 10}}, {"person", {20, 20, 30, 30}}};
  const auto m = aggregate({match(dets, gts, 0.5)});
  EXPECT_NEAR(m.map, 1.0, 1e-12);
  EXPECT_NEAR(m.micro_precision, 1.0, 1e-12);
  EXPECT_NEAR(m.micro_recall, 1.0, 1e-12);
  ASSERT_EQ(m.per_class.size(), 2u);
  // sorted by class name: car, person
  EXPECT_EQ(m.per_class[0].cls, "car");
  EXPECT_EQ(m.per_class[0].tp, 1);
  EXPECT_EQ(m.per_class[0].fn, 0);
}

TEST(Aggregate, MissedGtCountsAsFn) {
  std::vector<Det> dets{};
  std::vector<Gt> gts{{"car", {0, 0, 10, 10}}};
  const auto m = aggregate({match(dets, gts, 0.5)});
  ASSERT_EQ(m.per_class.size(), 1u);
  EXPECT_EQ(m.per_class[0].tp, 0);
  EXPECT_EQ(m.per_class[0].fn, 1);
  EXPECT_DOUBLE_EQ(m.per_class[0].recall, 0.0);
}

TEST(Aggregate, TwoClassMapAverages) {
  // car: perfect (ap=1). person: one FP, GT missed -> ap=0. map=0.5.
  std::vector<Det> dets{{"car", {0, 0, 10, 10}, 0.9}, {"person", {100, 100, 110, 110}, 0.7}};
  std::vector<Gt> gts{{"car", {0, 0, 10, 10}}, {"person", {0, 0, 10, 10}}};
  const auto m = aggregate({match(dets, gts, 0.5)});
  EXPECT_NEAR(m.map, 0.5, 1e-12);
}
