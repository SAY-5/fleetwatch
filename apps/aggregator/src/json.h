#ifndef FWAGG_JSON_H
#define FWAGG_JSON_H

#include <stdexcept>
#include <string>

#include "map.h"
#include "match.h"

namespace fwagg {

struct ParseError : std::runtime_error {
  explicit ParseError(const std::string& m) : std::runtime_error(m) {
  }
};

// The request the Python side sends over stdin.
struct Request {
  double iou_threshold;
  std::vector<std::vector<Det>> frame_dets;
  std::vector<std::vector<Gt>> frame_gts;
};

// Parse a FleetWatch aggregator request from a JSON string. Throws ParseError
// on malformed input (used by the fuzz test).
Request parse_request(const std::string& text);

// Serialise batch metrics as the canonical JSON response. Floating-point fields
// are written with 12 significant digits so Python can read them back exactly
// to within the documented 1e-9 parity tolerance.
std::string serialize_metrics(const BatchMetrics& m, double iou_threshold);

}  // namespace fwagg

#endif  // FWAGG_JSON_H
