#ifndef FWAGG_MAP_H
#define FWAGG_MAP_H

#include <string>
#include <vector>

#include "match.h"

namespace fwagg {

struct ClassMetric {
  std::string cls;
  long tp;
  long fp;
  long fn;
  double precision;
  double recall;
  double f1;
  double ap;
};

struct BatchMetrics {
  std::vector<ClassMetric> per_class;  // sorted by class name
  double map;                          // mean AP across classes
  double micro_precision;
  double micro_recall;
  double micro_f1;
};

// Aggregate a batch of per-frame match results into per-class and overall
// detection metrics at a single IoU threshold.
BatchMetrics aggregate(const std::vector<MatchResult>& frames);

}  // namespace fwagg

#endif  // FWAGG_MAP_H
