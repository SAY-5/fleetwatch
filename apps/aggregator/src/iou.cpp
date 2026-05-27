#include "iou.h"

#include <algorithm>

namespace fwagg {

double iou(const Box& a, const Box& b) {
  const double ix1 = std::max(a.x1, b.x1);
  const double iy1 = std::max(a.y1, b.y1);
  const double ix2 = std::min(a.x2, b.x2);
  const double iy2 = std::min(a.y2, b.y2);

  const double iw = ix2 - ix1;
  const double ih = iy2 - iy1;
  if (iw <= 0.0 || ih <= 0.0) {
    return 0.0;
  }
  const double inter = iw * ih;
  const double area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  const double area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  const double uni = area_a + area_b - inter;
  if (uni <= 0.0) {
    return 0.0;
  }
  return inter / uni;
}

}  // namespace fwagg
