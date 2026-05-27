#include "prcurve.h"

#include <algorithm>

namespace fwagg {

std::vector<PrPoint> pr_curve(std::vector<ScoredDet> scored, int n_gt) {
  // Sort by confidence descending; ties put TPs first for a reproducible curve.
  std::stable_sort(scored.begin(), scored.end(), [](const ScoredDet& a, const ScoredDet& b) {
    if (a.confidence != b.confidence) {
      return a.confidence > b.confidence;
    }
    return a.is_tp && !b.is_tp;
  });

  std::vector<PrPoint> curve;
  curve.reserve(scored.size());
  long tp = 0;
  long fp = 0;
  for (const auto& s : scored) {
    if (s.is_tp) {
      ++tp;
    } else {
      ++fp;
    }
    const double denom = static_cast<double>(tp + fp);
    const double precision = denom > 0.0 ? static_cast<double>(tp) / denom : 0.0;
    const double recall = n_gt > 0 ? static_cast<double>(tp) / static_cast<double>(n_gt) : 0.0;
    curve.push_back(PrPoint{s.confidence, precision, recall});
  }
  return curve;
}

double average_precision(const std::vector<PrPoint>& curve, int n_gt) {
  if (n_gt <= 0) {
    return 0.0;
  }
  if (curve.empty()) {
    return 0.0;
  }

  // Recalls and precisions padded with the (recall=0, precision=1) sentinel.
  std::vector<double> rec;
  std::vector<double> prec;
  rec.reserve(curve.size() + 2);
  prec.reserve(curve.size() + 2);
  rec.push_back(0.0);
  prec.push_back(1.0);
  for (const auto& p : curve) {
    rec.push_back(p.recall);
    prec.push_back(p.precision);
  }

  // Make precision monotonically non-increasing from the right.
  for (long i = static_cast<long>(prec.size()) - 2; i >= 0; --i) {
    prec[static_cast<size_t>(i)] =
        std::max(prec[static_cast<size_t>(i)], prec[static_cast<size_t>(i + 1)]);
  }

  // Integrate precision over the recall axis where recall increases.
  double ap = 0.0;
  for (size_t i = 1; i < rec.size(); ++i) {
    const double dr = rec[i] - rec[i - 1];
    if (dr > 0.0) {
      ap += dr * prec[i];
    }
  }
  return ap;
}

}  // namespace fwagg
