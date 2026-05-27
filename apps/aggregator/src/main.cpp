// FleetWatch aggregator entry point.
//
// Reads a single JSON request object from stdin, computes batch detection
// metrics (per-class precision/recall/F1/AP plus mAP and micro-averages) and
// writes the JSON response to stdout. See docs/aggregator-protocol.md.

#include <iostream>
#include <iterator>
#include <string>

#include "json.h"
#include "map.h"
#include "match.h"

int main() {
  std::ios::sync_with_stdio(false);
  const std::string input((std::istreambuf_iterator<char>(std::cin)),
                          std::istreambuf_iterator<char>());

  try {
    const fwagg::Request req = fwagg::parse_request(input);
    std::vector<fwagg::MatchResult> frames;
    frames.reserve(req.frame_dets.size());
    for (size_t i = 0; i < req.frame_dets.size(); ++i) {
      frames.push_back(fwagg::match(req.frame_dets[i], req.frame_gts[i], req.iou_threshold));
    }
    const fwagg::BatchMetrics metrics = fwagg::aggregate(frames);
    std::cout << fwagg::serialize_metrics(metrics, req.iou_threshold) << std::endl;
    return 0;
  } catch (const fwagg::ParseError& e) {
    std::cerr << "parse error: " << e.what() << std::endl;
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
