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

TEST(Json, AcceptsSubnormalNumbers) {
  // A subnormal value underflows toward zero but is still valid JSON; strtod
  // returns it without throwing, unlike std::stod.
  const auto req = parse_request(
      R"({"frames":[{"ground_truth":[{"class":"c","bbox":[0,2.225073858507e-311,1,1]}]}]})");
  ASSERT_EQ(req.frame_gts.size(), 1u);
  ASSERT_EQ(req.frame_gts[0].size(), 1u);
}

// Deterministic fuzz: every prefix and single-character mutation of a valid
// request must either parse or raise ParseError, never crash or hang.
TEST(Json, FuzzPrefixesAndMutationsNeverCrash) {
  const std::string valid =
      R"({"iou_threshold":0.5,"frames":[{"detections":[{"class":"car",)"
      R"("bbox":[0,0,10,10],"confidence":0.9}],"ground_truth":[{"class":"car",)"
      R"("bbox":[0,0,10,10]}]}]})";

  for (size_t cut = 0; cut <= valid.size(); ++cut) {
    const std::string prefix = valid.substr(0, cut);
    try {
      parse_request(prefix);
    } catch (const ParseError&) {
      // acceptable
    }
  }

  const char injects[] = {'{', '}', '[', ']', ',', ':', '"', 'x', '9', '\0', '-'};
  for (size_t pos = 0; pos < valid.size(); pos += 3) {
    for (char ch : injects) {
      std::string mutated = valid;
      mutated[pos] = ch;
      try {
        parse_request(mutated);
      } catch (const ParseError&) {
        // acceptable
      }
    }
  }
  SUCCEED();
}
