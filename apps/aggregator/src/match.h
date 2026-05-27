#ifndef FWAGG_MATCH_H
#define FWAGG_MATCH_H

#include <string>
#include <vector>

#include "iou.h"

namespace fwagg {

struct Det {
  std::string cls;
  Box box;
  double confidence;
};

struct Gt {
  std::string cls;
  Box box;
};

// Outcome of greedy matching for one detection, in confidence-descending order.
struct MatchedDet {
  std::string cls;
  double confidence;
  bool is_tp;  // matched a ground-truth box at or above the IoU threshold
};

struct MatchResult {
  std::vector<MatchedDet> dets;  // sorted by confidence desc, stable on ties
  // number of ground-truth boxes per class (the recall denominator)
  std::vector<std::pair<std::string, int>> gt_per_class;
};

// Greedy per-class matching identical to the Python reference:
//   1. detections sorted by confidence descending; ties broken by original
//      index ascending (stable),
//   2. each detection claims the highest-IoU unclaimed GT of the same class
//      whose IoU >= iou_threshold; that detection is a TP, otherwise a FP,
//   3. unclaimed GTs are false negatives.
MatchResult match(const std::vector<Det>& dets, const std::vector<Gt>& gts, double iou_threshold);

}  // namespace fwagg

#endif  // FWAGG_MATCH_H
