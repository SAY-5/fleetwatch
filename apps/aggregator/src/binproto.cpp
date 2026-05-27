#include "binproto.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace fwagg {

namespace {

class Cursor {
 public:
  explicit Cursor(const std::string& d) : d_(d), i_(0) {
  }

  void need(size_t n) const {
    if (i_ + n > d_.size()) {
      throw ParseError("binary request truncated");
    }
  }

  uint16_t u16() {
    need(2);
    uint16_t v;
    std::memcpy(&v, d_.data() + i_, 2);
    i_ += 2;
    return v;
  }

  uint32_t u32() {
    need(4);
    uint32_t v;
    std::memcpy(&v, d_.data() + i_, 4);
    i_ += 4;
    return v;
  }

  double f64() {
    need(8);
    double v;
    std::memcpy(&v, d_.data() + i_, 8);
    i_ += 8;
    return v;
  }

  std::string str(size_t n) {
    need(n);
    std::string s(d_.data() + i_, n);
    i_ += n;
    return s;
  }

  size_t pos() const {
    return i_;
  }
  size_t size() const {
    return d_.size();
  }

 private:
  const std::string& d_;
  size_t i_;
};

Box read_box(Cursor& c) {
  Box b;
  b.x1 = c.f64();
  b.y1 = c.f64();
  b.x2 = c.f64();
  b.y2 = c.f64();
  if (b.x2 <= b.x1 || b.y2 <= b.y1) {
    throw ParseError("bbox must satisfy x2 > x1 and y2 > y1");
  }
  return b;
}

}  // namespace

bool looks_binary(const std::string& data) {
  return data.size() >= 4 && data[0] == 'F' && data[1] == 'W' && data[2] == 'B' && data[3] == '1';
}

Request parse_binary(const std::string& data) {
  Cursor c(data);
  if (c.str(4) != "FWB1") {
    throw ParseError("bad magic");
  }
  Request req;
  req.iou_threshold = c.f64();
  if (req.iou_threshold < 0.0 || req.iou_threshold > 1.0) {
    throw ParseError("iou_threshold out of range");
  }

  const uint32_t n_classes = c.u32();
  std::vector<std::string> classes;
  classes.reserve(n_classes);
  for (uint32_t i = 0; i < n_classes; ++i) {
    const uint16_t len = c.u16();
    classes.push_back(c.str(len));
  }

  const auto class_name = [&](uint16_t id) -> const std::string& {
    if (id >= classes.size()) {
      throw ParseError("class id out of range");
    }
    return classes[id];
  };

  const uint32_t n_frames = c.u32();
  req.frame_dets.reserve(n_frames);
  req.frame_gts.reserve(n_frames);
  for (uint32_t f = 0; f < n_frames; ++f) {
    const uint32_t n_det = c.u32();
    const uint32_t n_gt = c.u32();
    std::vector<Det> dets;
    dets.reserve(n_det);
    for (uint32_t i = 0; i < n_det; ++i) {
      const uint16_t cid = c.u16();
      Det d;
      d.cls = class_name(cid);
      d.box = read_box(c);
      d.confidence = c.f64();
      dets.push_back(std::move(d));
    }
    std::vector<Gt> gts;
    gts.reserve(n_gt);
    for (uint32_t i = 0; i < n_gt; ++i) {
      const uint16_t cid = c.u16();
      Gt g;
      g.cls = class_name(cid);
      g.box = read_box(c);
      gts.push_back(std::move(g));
    }
    req.frame_dets.push_back(std::move(dets));
    req.frame_gts.push_back(std::move(gts));
  }

  if (c.pos() != c.size()) {
    throw ParseError("trailing bytes in binary request");
  }
  return req;
}

}  // namespace fwagg
