#include <gtest/gtest.h>

#include "prcurve.h"

using fwagg::average_precision;
using fwagg::pr_curve;
using fwagg::ScoredDet;

TEST(PrCurve, PerfectDetectionsApIsOne) {
  std::vector<ScoredDet> s{{0.9, true}, {0.8, true}, {0.7, true}};
  const auto curve = pr_curve(s, 3);
  EXPECT_NEAR(average_precision(curve, 3), 1.0, 1e-12);
}

TEST(PrCurve, AllFalsePositivesApIsZero) {
  std::vector<ScoredDet> s{{0.9, false}, {0.8, false}};
  const auto curve = pr_curve(s, 2);
  EXPECT_NEAR(average_precision(curve, 2), 0.0, 1e-12);
}

TEST(PrCurve, HandComputedMixed) {
  // 2 GT. detections by confidence: TP, FP, TP.
  // step 1: tp=1 fp=0 -> p=1.0 r=0.5
  // step 2: tp=1 fp=1 -> p=0.5 r=0.5
  // step 3: tp=2 fp=1 -> p=2/3 r=1.0
  // envelope from right: [1.0(sentinel r0), then max stuff]
  // points (r,p): (0,1),(0.5,1.0),(0.5,0.5),(1.0,0.6667)
  // monotone precision from right: 0.6667,0.6667,0.6667,0.6667? compute:
  //   index3 p=0.6667; index2 p=max(0.5,0.6667)=0.6667; index1
  //   p=max(1.0,0.6667)=1.0; index0 p=max(1.0,1.0)=1.0
  // integrate over dr>0: at r 0->0.5 dr=0.5 p=1.0 => 0.5; r0.5->1.0 dr=0.5
  //   p=0.6667 => 0.3333. total = 0.8333.
  std::vector<ScoredDet> s{{0.9, true}, {0.8, false}, {0.7, true}};
  const auto curve = pr_curve(s, 2);
  EXPECT_NEAR(average_precision(curve, 2), 0.8333333333333333, 1e-9);
}

TEST(PrCurve, NoGroundTruthApIsZero) {
  std::vector<ScoredDet> s{{0.9, false}};
  const auto curve = pr_curve(s, 0);
  EXPECT_DOUBLE_EQ(average_precision(curve, 0), 0.0);
}
