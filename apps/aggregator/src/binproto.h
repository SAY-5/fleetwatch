#ifndef FWAGG_BINPROTO_H
#define FWAGG_BINPROTO_H

#include <string>

#include "json.h"

// Compact little-endian binary request protocol for fleet-scale batches.
//
// The JSON protocol is convenient but its byte volume and text parsing dominate
// the runtime on multi-hundred-MB batches. The binary protocol carries the same
// information in roughly a fifth of the bytes with no text parsing.
//
// Layout (all little-endian):
//   magic        : 4 bytes  "FWB1"
//   iou_threshold: f64
//   n_classes    : u32, then n_classes [u16 len][len bytes] class names
//   n_frames     : u32
//   per frame    : u32 n_det, u32 n_gt
//                  n_det  * { u16 cls_id, f64 x1,y1,x2,y2, f64 conf }
//                  n_gt   * { u16 cls_id, f64 x1,y1,x2,y2 }
//
// Box validity (x2 > x1, y2 > y1) is checked while decoding, matching the JSON
// path, so the fuzz contract holds here too.

namespace fwagg {

bool looks_binary(const std::string& data);

// Decode a binary request. Throws ParseError on malformed input.
Request parse_binary(const std::string& data);

}  // namespace fwagg

#endif  // FWAGG_BINPROTO_H
