#include <gtest/gtest.h>

#include "match.h"

using fwagg::Det;
using fwagg::Gt;
using fwagg::match;

TEST(Match, PerfectSingleMatchIsTp) {
  std::vector<Det> dets{{"car", {0, 0, 10, 10}, 0.9}};
  std::vector<Gt> gts{{"car", {0, 0, 10, 10}}};
  const auto r = match(dets, gts, 0.5);
  ASSERT_EQ(r.dets.size(), 1u);
  EXPECT_TRUE(r.dets[0].is_tp);
  ASSERT_EQ(r.gt_per_class.size(), 1u);
  EXPECT_EQ(r.gt_per_class[0].second, 1);
}

TEST(Match, ClassMismatchIsFp) {
  std::vector<Det> dets{{"car", {0, 0, 10, 10}, 0.9}};
  std::vector<Gt> gts{{"person", {0, 0, 10, 10}}};
  const auto r = match(dets, gts, 0.5);
  EXPECT_FALSE(r.dets[0].is_tp);
}

TEST(Match, BelowThresholdIsFp) {
  // iou = 1/3 < 0.5 -> false positive.
  std::vector<Det> dets{{"car", {5, 0, 15, 10}, 0.9}};
  std::vector<Gt> gts{{"car", {0, 0, 10, 10}}};
  const auto r = match(dets, gts, 0.5);
  EXPECT_FALSE(r.dets[0].is_tp);
}

TEST(Match, HigherConfidenceClaimsGtFirst) {
  // Two detections overlap one GT; the higher-confidence one wins, the other
  // becomes a false positive.
  std::vector<Det> dets{{"car", {0, 0, 10, 10}, 0.6}, {"car", {0, 0, 10, 10}, 0.9}};
  std::vector<Gt> gts{{"car", {0, 0, 10, 10}}};
  const auto r = match(dets, gts, 0.5);
  // results are confidence-descending
  EXPECT_DOUBLE_EQ(r.dets[0].confidence, 0.9);
  EXPECT_TRUE(r.dets[0].is_tp);
  EXPECT_DOUBLE_EQ(r.dets[1].confidence, 0.6);
  EXPECT_FALSE(r.dets[1].is_tp);
}
