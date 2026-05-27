#include "map.h"

#include <algorithm>
#include <map>

#include "prcurve.h"

namespace fwagg {

namespace {

double safe_div(long num, long den) {
  return den > 0 ? static_cast<double>(num) / static_cast<double>(den) : 0.0;
}

double f1_of(double p, double r) {
  const double s = p + r;
  return s > 0.0 ? 2.0 * p * r / s : 0.0;
}

}  // namespace

BatchMetrics aggregate(const std::vector<MatchResult>& frames) {
  // Per-class accumulators. Scored detections are collected in frame order,
  // then within-frame in the confidence-descending order produced by match(),
  // which fixes the accumulation order for parity with Python.
  std::map<std::string, std::vector<ScoredDet>> scored;
  std::map<std::string, int> gt_count;

  for (const auto& fr : frames) {
    for (const auto& d : fr.dets) {
      scored[d.cls].push_back(ScoredDet{d.confidence, d.is_tp});
    }
    for (const auto& gc : fr.gt_per_class) {
      gt_count[gc.first] += gc.second;
    }
  }

  // Union of classes that appear either as a detection or as ground truth.
  std::vector<std::string> classes;
  for (const auto& kv : scored) {
    classes.push_back(kv.first);
  }
  for (const auto& kv : gt_count) {
    if (scored.find(kv.first) == scored.end()) {
      classes.push_back(kv.first);
    }
  }
  std::sort(classes.begin(), classes.end());
  classes.erase(std::unique(classes.begin(), classes.end()), classes.end());

  BatchMetrics out;
  double ap_sum = 0.0;
  long total_tp = 0;
  long total_fp = 0;
  long total_fn = 0;

  for (const auto& c : classes) {
    const auto it = scored.find(c);
    const std::vector<ScoredDet> dets = it != scored.end() ? it->second : std::vector<ScoredDet>{};
    const int n_gt = gt_count.count(c) ? gt_count[c] : 0;

    long tp = 0;
    long fp = 0;
    for (const auto& s : dets) {
      if (s.is_tp) {
        ++tp;
      } else {
        ++fp;
      }
    }
    const long fn = static_cast<long>(n_gt) - tp;

    const double precision = safe_div(tp, tp + fp);
    const double recall = safe_div(tp, n_gt);
    const double f1 = f1_of(precision, recall);

    const auto curve = pr_curve(dets, n_gt);
    const double ap = average_precision(curve, n_gt);

    out.per_class.push_back(ClassMetric{c, tp, fp, fn, precision, recall, f1, ap});
    ap_sum += ap;
    total_tp += tp;
    total_fp += fp;
    total_fn += fn;
  }

  out.map = classes.empty() ? 0.0 : ap_sum / static_cast<double>(classes.size());
  out.micro_precision = safe_div(total_tp, total_tp + total_fp);
  out.micro_recall = safe_div(total_tp, total_tp + total_fn);
  out.micro_f1 = f1_of(out.micro_precision, out.micro_recall);
  return out;
}

}  // namespace fwagg
