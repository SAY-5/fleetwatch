#ifndef FWAGG_IOU_H
#define FWAGG_IOU_H

namespace fwagg {

// Axis-aligned bounding box in pixel coordinates, x2 > x1 and y2 > y1.
struct Box {
  double x1;
  double y1;
  double x2;
  double y2;
};

// Intersection-over-union of two boxes. Returns 0.0 when they do not overlap.
// Symmetric in its arguments.
double iou(const Box& a, const Box& b);

}  // namespace fwagg

#endif  // FWAGG_IOU_H
