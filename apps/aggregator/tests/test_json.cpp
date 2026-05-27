#include <gtest/gtest.h>

#include <string>

#include "json.h"

using fwagg::parse_request;
using fwagg::ParseError;

TEST(Json, ParsesMinimalRequest) {
  const std::string text = R"({"iou_threshold":0.5,"frames":[
    {"detections":[{"class":"car","bbox":[0,0,10,10],"confidence":0.9}],
     "ground_truth":[{"class":"car","bbox":[0,0,10,10]}]}]})";
  const auto req = parse_request(text);
  EXPECT_DOUBLE_EQ(req.iou_threshold, 0.5);
  ASSERT_EQ(req.frame_dets.size(), 1u);
  ASSERT_EQ(req.frame_dets[0].size(), 1u);
  EXPECT_EQ(req.frame_dets[0][0].cls, "car");
  EXPECT_DOUBLE_EQ(req.frame_dets[0][0].confidence, 0.9);
  ASSERT_EQ(req.frame_gts[0].size(), 1u);
}

TEST(Json, EmptyFramesAllowed) {
  const auto req = parse_request(R"({"frames":[]})");
  EXPECT_EQ(req.frame_dets.size(), 0u);
}

TEST(Json, RejectsMissingFrames) {
  EXPECT_THROW(parse_request(R"({"iou_threshold":0.5})"), ParseError);
}

TEST(Json, RejectsTruncatedInput) {
  EXPECT_THROW(parse_request(R"({"frames":[{"detections":)"), ParseError);
}

TEST(Json, RejectsBadBox) {
  EXPECT_THROW(parse_request(R"({"frames":[{"detections":[{"class":"c","bbox":[10,10,0,0]}]}]})"),
               ParseError);
}

TEST(Json, RejectsTrailingGarbage) {
  EXPECT_THROW(parse_request(R"({"frames":[]}xyz)"), ParseError);
}
