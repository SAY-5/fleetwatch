// FleetWatch aggregator entry point.
//
// Reads a single JSON request object from stdin, computes batch detection
// metrics (per-class precision/recall/F1/AP plus mAP and micro-averages) and
// writes the JSON response to stdout. See docs/aggregator-protocol.md.
//
// When FWAGG_TIMING=1 is set, the parse and compute durations (microseconds)
// are written to stderr as "timing parse_us=<n> compute_us=<n>". This lets the
// benchmark separate the C++ compute throughput from the JSON transport cost.

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "binproto.h"
#include "json.h"
#include "map.h"
#include "match.h"

namespace {
long long now_us() {
  using namespace std::chrono;
  return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

// Read all of stdin into a string with large block reads. The iostream and
// istreambuf_iterator paths are both O(n) with heavy per-byte overhead and
// dominate the runtime on multi-hundred-MB payloads; a raw read loop does not.
std::string read_stdin() {
  std::string out;
  constexpr size_t kChunk = 1 << 20;  // 1 MiB
  char buffer[kChunk];
  ssize_t n;
  while ((n = ::read(STDIN_FILENO, buffer, kChunk)) > 0) {
    out.append(buffer, static_cast<size_t>(n));
  }
  return out;
}
}  // namespace

int main() {
  const std::string input = read_stdin();
  const bool timing = std::getenv("FWAGG_TIMING") != nullptr;

  try {
    const long long t0 = now_us();
    const fwagg::Request req =
        fwagg::looks_binary(input) ? fwagg::parse_binary(input) : fwagg::parse_request(input);
    const long long t1 = now_us();

    std::vector<fwagg::MatchResult> frames;
    frames.reserve(req.frame_dets.size());
    for (size_t i = 0; i < req.frame_dets.size(); ++i) {
      frames.push_back(fwagg::match(req.frame_dets[i], req.frame_gts[i], req.iou_threshold));
    }
    const fwagg::BatchMetrics metrics = fwagg::aggregate(frames);
    const long long t2 = now_us();

    std::cout << fwagg::serialize_metrics(metrics, req.iou_threshold) << std::endl;
    if (timing) {
      std::cerr << "timing parse_us=" << (t1 - t0) << " compute_us=" << (t2 - t1) << std::endl;
    }
    return 0;
  } catch (const fwagg::ParseError& e) {
    std::cerr << "parse error: " << e.what() << std::endl;
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
