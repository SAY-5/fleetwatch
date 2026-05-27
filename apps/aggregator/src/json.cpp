#include "json.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>

namespace fwagg {

namespace {

// A minimal recursive-descent JSON reader sufficient for the request schema.
// It rejects malformed input by throwing ParseError, which makes it a clean
// target for the fuzz test on the C++ side.
class Reader {
 public:
  explicit Reader(const std::string& s) : s_(s), i_(0) {
  }

  void skip_ws() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) {
      ++i_;
    }
  }

  char peek() {
    skip_ws();
    if (i_ >= s_.size()) {
      throw ParseError("unexpected end of input");
    }
    return s_[i_];
  }

  char get() {
    const char c = peek();
    ++i_;
    return c;
  }

  void expect(char c) {
    if (get() != c) {
      throw ParseError(std::string("expected '") + c + "'");
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (i_ < s_.size()) {
      const char c = s_[i_++];
      if (c == '"') {
        return out;
      }
      if (c == '\\') {
        if (i_ >= s_.size()) {
          throw ParseError("bad escape");
        }
        const char e = s_[i_++];
        switch (e) {
          case '"':
            out.push_back('"');
            break;
          case '\\':
            out.push_back('\\');
            break;
          case '/':
            out.push_back('/');
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 't':
            out.push_back('\t');
            break;
          case 'r':
            out.push_back('\r');
            break;
          case 'b':
            out.push_back('\b');
            break;
          case 'f':
            out.push_back('\f');
            break;
          case 'u': {
            if (i_ + 4 > s_.size()) {
              throw ParseError("bad \\u escape");
            }
            i_ += 4;  // ASCII-only protocol; skip the code unit
            out.push_back('?');
            break;
          }
          default:
            throw ParseError("bad escape char");
        }
      } else {
        out.push_back(c);
      }
    }
    throw ParseError("unterminated string");
  }

  double parse_number() {
    skip_ws();
    const size_t start = i_;
    if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) {
      ++i_;
    }
    bool any = false;
    while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '.' ||
                              s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-')) {
      any = true;
      ++i_;
    }
    if (!any) {
      throw ParseError("expected number");
    }
    try {
      return std::stod(s_.substr(start, i_ - start));
    } catch (...) {
      throw ParseError("invalid number");
    }
  }

  void parse_box(Box& b) {
    expect('[');
    b.x1 = parse_number();
    expect(',');
    b.y1 = parse_number();
    expect(',');
    b.x2 = parse_number();
    expect(',');
    b.y2 = parse_number();
    expect(']');
    if (b.x2 <= b.x1 || b.y2 <= b.y1) {
      throw ParseError("bbox must satisfy x2 > x1 and y2 > y1");
    }
  }

  void at_end() {
    skip_ws();
    if (i_ != s_.size()) {
      throw ParseError("trailing characters");
    }
  }

 private:
  const std::string& s_;
  size_t i_;
};

}  // namespace

Request parse_request(const std::string& text) {
  Reader r(text);
  Request req;
  req.iou_threshold = 0.5;
  bool have_frames = false;

  r.expect('{');
  if (r.peek() == '}') {
    r.get();
    throw ParseError("missing frames");
  }
  while (true) {
    const std::string key = r.parse_string();
    r.expect(':');
    if (key == "iou_threshold") {
      req.iou_threshold = r.parse_number();
      if (req.iou_threshold < 0.0 || req.iou_threshold > 1.0) {
        throw ParseError("iou_threshold out of range");
      }
    } else if (key == "frames") {
      have_frames = true;
      r.expect('[');
      if (r.peek() != ']') {
        while (true) {
          std::vector<Det> dets;
          std::vector<Gt> gts;
          r.expect('{');
          if (r.peek() != '}') {
            while (true) {
              const std::string fkey = r.parse_string();
              r.expect(':');
              if (fkey == "detections") {
                r.expect('[');
                if (r.peek() != ']') {
                  while (true) {
                    Det d;
                    d.confidence = 1.0;
                    r.expect('{');
                    while (true) {
                      const std::string dk = r.parse_string();
                      r.expect(':');
                      if (dk == "class") {
                        d.cls = r.parse_string();
                      } else if (dk == "bbox") {
                        r.parse_box(d.box);
                      } else if (dk == "confidence") {
                        d.confidence = r.parse_number();
                      } else {
                        throw ParseError("unknown detection field");
                      }
                      if (r.peek() == ',') {
                        r.get();
                        continue;
                      }
                      break;
                    }
                    r.expect('}');
                    dets.push_back(d);
                    if (r.peek() == ',') {
                      r.get();
                      continue;
                    }
                    break;
                  }
                }
                r.expect(']');
              } else if (fkey == "ground_truth") {
                r.expect('[');
                if (r.peek() != ']') {
                  while (true) {
                    Gt g;
                    r.expect('{');
                    while (true) {
                      const std::string gk = r.parse_string();
                      r.expect(':');
                      if (gk == "class") {
                        g.cls = r.parse_string();
                      } else if (gk == "bbox") {
                        r.parse_box(g.box);
                      } else {
                        throw ParseError("unknown gt field");
                      }
                      if (r.peek() == ',') {
                        r.get();
                        continue;
                      }
                      break;
                    }
                    r.expect('}');
                    gts.push_back(g);
                    if (r.peek() == ',') {
                      r.get();
                      continue;
                    }
                    break;
                  }
                }
                r.expect(']');
              } else {
                throw ParseError("unknown frame field");
              }
              if (r.peek() == ',') {
                r.get();
                continue;
              }
              break;
            }
          }
          r.expect('}');
          req.frame_dets.push_back(std::move(dets));
          req.frame_gts.push_back(std::move(gts));
          if (r.peek() == ',') {
            r.get();
            continue;
          }
          break;
        }
      }
      r.expect(']');
    } else {
      throw ParseError("unknown top-level field");
    }
    if (r.peek() == ',') {
      r.get();
      continue;
    }
    break;
  }
  r.expect('}');
  r.at_end();
  if (!have_frames) {
    throw ParseError("missing frames");
  }
  return req;
}

namespace {

std::string num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.12g", v);
  return std::string(buf);
}

std::string jstr(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

}  // namespace

std::string serialize_metrics(const BatchMetrics& m, double iou_threshold) {
  std::ostringstream os;
  os << "{";
  os << "\"iou_threshold\":" << num(iou_threshold) << ",";
  os << "\"map\":" << num(m.map) << ",";
  os << "\"micro_precision\":" << num(m.micro_precision) << ",";
  os << "\"micro_recall\":" << num(m.micro_recall) << ",";
  os << "\"micro_f1\":" << num(m.micro_f1) << ",";
  os << "\"per_class\":[";
  for (size_t i = 0; i < m.per_class.size(); ++i) {
    const auto& c = m.per_class[i];
    if (i) {
      os << ",";
    }
    os << "{";
    os << "\"class\":" << jstr(c.cls) << ",";
    os << "\"tp\":" << c.tp << ",";
    os << "\"fp\":" << c.fp << ",";
    os << "\"fn\":" << c.fn << ",";
    os << "\"precision\":" << num(c.precision) << ",";
    os << "\"recall\":" << num(c.recall) << ",";
    os << "\"f1\":" << num(c.f1) << ",";
    os << "\"ap\":" << num(c.ap);
    os << "}";
  }
  os << "]}";
  return os.str();
}

}  // namespace fwagg
