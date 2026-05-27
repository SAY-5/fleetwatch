#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "binproto.h"

using fwagg::looks_binary;
using fwagg::parse_binary;
using fwagg::ParseError;

namespace {

void put_u16(std::string& s, uint16_t v) {
  char b[2];
  std::memcpy(b, &v, 2);
  s.append(b, 2);
}
void put_u32(std::string& s, uint32_t v) {
  char b[4];
  std::memcpy(b, &v, 4);
  s.append(b, 4);
}
void put_f64(std::string& s, double v) {
  char b[8];
  std::memcpy(b, &v, 8);
  s.append(b, 8);
}

// One frame, one detection and one ground truth of class "car".
std::string valid_request() {
  std::string s = "FWB1";
  put_f64(s, 0.5);  // iou_threshold
  put_u32(s, 1);    // n_classes
  put_u16(s, 3);    // len("car")
  s += "car";
  put_u32(s, 1);    // n_frames
  put_u32(s, 1);    // n_det
  put_u32(s, 1);    // n_gt
  put_u16(s, 0);    // det cls id
  put_f64(s, 0);    // x1
  put_f64(s, 0);    // y1
  put_f64(s, 10);   // x2
  put_f64(s, 10);   // y2
  put_f64(s, 0.9);  // conf
  put_u16(s, 0);    // gt cls id
  put_f64(s, 0);
  put_f64(s, 0);
  put_f64(s, 10);
  put_f64(s, 10);
  return s;
}

}  // namespace

TEST(BinProto, DetectsMagic) {
  EXPECT_TRUE(looks_binary("FWB1...."));
  EXPECT_FALSE(looks_binary("{\"frames\":[]}"));
  EXPECT_FALSE(looks_binary("FW"));
}

TEST(BinProto, ParsesValidRequest) {
  const auto req = parse_binary(valid_request());
  EXPECT_DOUBLE_EQ(req.iou_threshold, 0.5);
  ASSERT_EQ(req.frame_dets.size(), 1u);
  ASSERT_EQ(req.frame_dets[0].size(), 1u);
  EXPECT_EQ(req.frame_dets[0][0].cls, "car");
  EXPECT_DOUBLE_EQ(req.frame_dets[0][0].confidence, 0.9);
  ASSERT_EQ(req.frame_gts[0].size(), 1u);
}

TEST(BinProto, RejectsBadMagic) {
  EXPECT_THROW(parse_binary("XXXXyyyy"), ParseError);
}

TEST(BinProto, RejectsTruncated) {
  std::string s = valid_request();
  s.resize(s.size() - 4);
  EXPECT_THROW(parse_binary(s), ParseError);
}

TEST(BinProto, RejectsTrailingBytes) {
  std::string s = valid_request();
  s += "junk";
  EXPECT_THROW(parse_binary(s), ParseError);
}

TEST(BinProto, RejectsClassIdOutOfRange) {
  std::string s = "FWB1";
  put_f64(s, 0.5);
  put_u32(s, 0);  // zero classes
  put_u32(s, 1);  // one frame
  put_u32(s, 1);  // one det
  put_u32(s, 0);  // no gt
  put_u16(s, 5);  // class id 5 (out of range)
  put_f64(s, 0);
  put_f64(s, 0);
  put_f64(s, 10);
  put_f64(s, 10);
  put_f64(s, 0.9);
  EXPECT_THROW(parse_binary(s), ParseError);
}
