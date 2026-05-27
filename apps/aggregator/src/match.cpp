#include "match.h"

#include <algorithm>
#include <map>
#include <numeric>

namespace fwagg {

MatchResult match(const std::vector<Det>& dets, const std::vector<Gt>& gts, double iou_threshold) {
  MatchResult res;

  // Stable sort of detection indices by confidence descending, ties by index.
  std::vector<size_t> order(dets.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return dets[a].confidence > dets[b].confidence; });

  std::vector<bool> gt_claimed(gts.size(), false);

  for (size_t oi : order) {
    const Det& d = dets[oi];
    double best_iou = iou_threshold;  // strictly must meet the threshold
    long best_gt = -1;
    for (size_t gi = 0; gi < gts.size(); ++gi) {
      if (gt_claimed[gi] || gts[gi].cls != d.cls) {
        continue;
      }
      const double v = iou(d.box, gts[gi].box);
      if (v >= best_iou) {
        best_iou = v;
        best_gt = static_cast<long>(gi);
      }
    }
    const bool is_tp = best_gt >= 0;
    if (is_tp) {
      gt_claimed[static_cast<size_t>(best_gt)] = true;
    }
    res.dets.push_back(MatchedDet{d.cls, d.confidence, is_tp});
  }

  // Count GT boxes per class, ordered by first appearance for stable output.
  std::map<std::string, int> counts;
  std::vector<std::string> seen;
  for (const auto& g : gts) {
    if (counts.find(g.cls) == counts.end()) {
      seen.push_back(g.cls);
    }
    counts[g.cls] += 1;
  }
  std::sort(seen.begin(), seen.end());
  for (const auto& c : seen) {
    res.gt_per_class.emplace_back(c, counts[c]);
  }
  return res;
}

}  // namespace fwagg
