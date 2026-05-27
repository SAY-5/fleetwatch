#ifndef FWAGG_PRCURVE_H
#define FWAGG_PRCURVE_H

#include <string>
#include <vector>

namespace fwagg {

// A single matched detection contributing to a class's PR curve.
struct ScoredDet {
  double confidence;
  bool is_tp;
};

struct PrPoint {
  double confidence;
  double precision;
  double recall;
};

// Build the PR curve for one class from scored detections (any order) and the
// total ground-truth count. Detections are sorted by confidence descending,
// ties broken by treating true positives before false positives so the curve
// is reproducible. Returns one point per detection in that order.
std::vector<PrPoint> pr_curve(std::vector<ScoredDet> scored, int n_gt);

// Average precision: area under the PR curve using the all-points (continuous)
// method with a monotonically non-increasing precision envelope. Matches the
// Python reference exactly.
double average_precision(const std::vector<PrPoint>& curve, int n_gt);

}  // namespace fwagg

#endif  // FWAGG_PRCURVE_H
