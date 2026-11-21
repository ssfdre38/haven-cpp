#pragma once

#include <string>
#include <vector>
#include <map>
#include <variant>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace haven {

// Strict, depth-limited RFC 8259 recursive-descent JSON parser.
// Zero external dependencies, bounded recursion depth, full UTF-8 \uXXXX + surrogate pair decoding.
class HavenJson {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    using Array = std::vector<HavenJson>;
    using Object = std::map<std::string, HavenJson>;

    HavenJson() : type_(Type::Null), data_(std::monostate{}) {}
    explicit HavenJson(bool b) : type_(Type::Bool), data_(b) {}
    explicit HavenJson(double n, std::string raw = "") : type_(Type::Number), data_(n), raw_number_(std::move(raw)) {}
    explicit HavenJson(std::string s) : type_(Type::String), data_(std::move(s)) {}
    explicit HavenJson(Array a) : type_(Type::Array), data_(std::move(a)) {}
    explicit HavenJson(Object o) : type_(Type::Object), data_(std::move(o)) {}

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool as_bool(bool fallback = false) const {
        return type_ == Type::Bool ? std::get<bool>(data_) : fallback;
    }

    double as_double(double fallback = 0.0) const {
        return type_ == Type::Number ? std::get<double>(data_) : fallback;
    }

    const std::string& as_string() const {
        static const std::string empty;
        return type_ == Type::String ? std::get<std::string>(data_) : empty;
    }

    const Array& as_array() const {
        static const Array empty;
        return type_ == Type::Array ? std::get<Array>(data_) : empty;
    }

    const Object& as_object() const {
        static const Object empty;
        return type_ == Type::Object ? std::get<Object>(data_) : empty;
    }

    const HavenJson* find(const std::string& key) const {
        if (type_ != Type::Object) return nullptr;
        const auto& obj = std::get<Object>(data_);
        auto it = obj.find(key);
        return (it != obj.end()) ? &it->second : nullptr;
    }

    std::string get_string(const std::string& key, const std::string& fallback = "") const {
        const HavenJson* v = find(key);
        if (v && v->is_string()) return v->as_string();
        return fallback;
    }

    // Returns string representation for string, number, or bool scalars (empty for null/object/array/missing)
    std::string get_scalar_string(const std::string& key) const {
        const HavenJson* v = find(key);
        if (!v) return "";
        if (v->is_string()) return v->as_string();
        if (v->is_number()) return v->raw_number_.empty() ? std::to_string(v->as_double()) : v->raw_number_;
        if (v->is_bool()) return v->as_bool() ? "true" : "false";
        return "";
    }

    static bool parse(const std::string& input, HavenJson& out, size_t max_depth = 32) {
        constexpr size_t MAX_JSON_BYTES = 64ULL * 1024 * 1024;
        if (input.empty() || input.size() > MAX_JSON_BYTES) return false;
        Parser p(input, max_depth);
        p.skip_ws();
        if (!p.parse_value(out, 0)) return false;
        p.skip_ws();
        return p.pos_ == p.src_.size();
    }

private:
    Type type_;
    std::variant<std::monostate, bool, double, std::string, Array, Object> data_;
    std::string raw_number_;

    struct Parser {
        const std::string& src_;
        size_t pos_ = 0;
        size_t max_depth_;

        Parser(const std::string& s, size_t md) : src_(s), max_depth_(md) {}

        void skip_ws() {
            while (pos_ < src_.size()) {
                char c = src_[pos_];
                if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                    ++pos_;
                } else {
                    break;
                }
            }
        }

        static void append_utf8(std::string& out, uint32_t cp) {
            if (cp <= 0x7F) {
                out += static_cast<char>(cp);
            } else if (cp <= 0x7FF) {
                out += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            } else if (cp <= 0xFFFF) {
                out += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            } else if (cp <= 0x10FFFF) {
                out += static_cast<char>(0xF0 | ((cp >> 18) & 0x07));
                out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            }
        }

        bool parse_hex4(uint32_t& out) {
            if (pos_ + 4 > src_.size()) return false;
            uint32_t v = 0;
            for (int i = 0; i < 4; ++i) {
                char c = src_[pos_++];
                v <<= 4;
                if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
                else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
                else return false;
            }
            out = v;
            return true;
        }

        bool parse_string(std::string& out) {
            if (pos_ >= src_.size() || src_[pos_] != '"') return false;
            ++pos_;
            out.clear();
            while (pos_ < src_.size()) {
                unsigned char uc = static_cast<unsigned char>(src_[pos_++]);
                if (uc == '"') return true;
                if (uc < 0x20) return false; // Unescaped control characters forbidden by RFC 8259
                if (uc == '\\') {
                    if (pos_ >= src_.size()) return false;
                    char esc = src_[pos_++];
                    switch (esc) {
                        case '"':  out += '"'; break;
                        case '\\': out += '\\'; break;
                        case '/':  out += '/'; break;
                        case 'b':  out += '\b'; break;
                        case 'f':  out += '\f'; break;
                        case 'n':  out += '\n'; break;
                        case 'r':  out += '\r'; break;
                        case 't':  out += '\t'; break;
                        case 'u': {
                            uint32_t cp = 0;
                            if (!parse_hex4(cp)) return false;
                            if (cp >= 0xD800 && cp <= 0xDBFF) {
                                // UTF-16 high surrogate: expect low surrogate
                                if (pos_ + 2 <= src_.size() && src_[pos_] == '\\' && src_[pos_ + 1] == 'u') {
                                    pos_ += 2;
                                    uint32_t low = 0;
                                    if (!parse_hex4(low)) return false;
                                    if (low >= 0xDC00 && low <= 0xDFFF) {
                                        cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                                    } else {
                                        return false;
                                    }
                                } else {
                                    return false;
                                }
                            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                                return false; // Lone low surrogate
                            }
                            append_utf8(out, cp);
                            break;
                        }
                        default:
                            return false;
                    }
                } else {
                    out += static_cast<char>(uc);
                }
            }
            return false; // Unterminated string
        }

        bool parse_number(HavenJson& out) {
            size_t start = pos_;
            if (pos_ < src_.size() && src_[pos_] == '-') ++pos_;
            if (pos_ >= src_.size()) return false;
            if (src_[pos_] == '0') {
                ++pos_;
            } else if (src_[pos_] >= '1' && src_[pos_] <= '9') {
                while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') ++pos_;
            } else {
                return false;
            }
            if (pos_ < src_.size() && src_[pos_] == '.') {
                ++pos_;
                if (pos_ >= src_.size() || src_[pos_] < '0' || src_[pos_] > '9') return false;
                while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') ++pos_;
            }
            if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
                ++pos_;
                if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) ++pos_;
                if (pos_ >= src_.size() || src_[pos_] < '0' || src_[pos_] > '9') return false;
                while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') ++pos_;
            }
            std::string raw = src_.substr(start, pos_ - start);
            char* endptr = nullptr;
            double val = std::strtod(raw.c_str(), &endptr);
            if (!endptr || endptr != raw.c_str() + raw.size() || !std::isfinite(val)) {
                return false;
            }
            out = HavenJson(val, std::move(raw));
            return true;
        }

        bool parse_array(HavenJson& out, size_t depth) {
            if (depth >= max_depth_) return false;
            ++pos_; // skip '['
            Array arr;
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == ']') {
                ++pos_;
                out = HavenJson(std::move(arr));
                return true;
            }
            constexpr size_t MAX_ELEMENTS = 100000;
            while (pos_ < src_.size()) {
                if (arr.size() >= MAX_ELEMENTS) return false;
                HavenJson elem;
                skip_ws();
                if (!parse_value(elem, depth + 1)) return false;
                arr.push_back(std::move(elem));
                skip_ws();
                if (pos_ >= src_.size()) return false;
                if (src_[pos_] == ']') {
                    ++pos_;
                    out = HavenJson(std::move(arr));
                    return true;
                }
                if (src_[pos_] != ',') return false;
                ++pos_;
            }
            return false;
        }

        bool parse_object(HavenJson& out, size_t depth) {
            if (depth >= max_depth_) return false;
            ++pos_; // skip '{'
            Object obj;
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == '}') {
                ++pos_;
                out = HavenJson(std::move(obj));
                return true;
            }
            constexpr size_t MAX_FIELDS = 100000;
            while (pos_ < src_.size()) {
                if (obj.size() >= MAX_FIELDS) return false;
                skip_ws();
                std::string key;
                if (!parse_string(key)) return false;
                skip_ws();
                if (pos_ >= src_.size() || src_[pos_] != ':') return false;
                ++pos_;
                skip_ws();
                HavenJson val;
                if (!parse_value(val, depth + 1)) return false;
                obj[std::move(key)] = std::move(val);
                skip_ws();
                if (pos_ >= src_.size()) return false;
                if (src_[pos_] == '}') {
                    ++pos_;
                    out = HavenJson(std::move(obj));
                    return true;
                }
                if (src_[pos_] != ',') return false;
                ++pos_;
            }
            return false;
        }

        bool parse_value(HavenJson& out, size_t depth) {
            if (depth > max_depth_ || pos_ >= src_.size()) return false;
            char c = src_[pos_];
            if (c == '"') {
                std::string s;
                if (!parse_string(s)) return false;
                out = HavenJson(std::move(s));
                return true;
            }
            if (c == '{') return parse_object(out, depth);
            if (c == '[') return parse_array(out, depth);
            if (c == 't') {
                if (pos_ + 4 <= src_.size() && src_.compare(pos_, 4, "true") == 0) {
                    pos_ += 4;
                    out = HavenJson(true);
                    return true;
                }
                return false;
            }
            if (c == 'f') {
                if (pos_ + 5 <= src_.size() && src_.compare(pos_, 5, "false") == 0) {
                    pos_ += 5;
                    out = HavenJson(false);
                    return true;
                }
                return false;
            }
            if (c == 'n') {
                if (pos_ + 4 <= src_.size() && src_.compare(pos_, 4, "null") == 0) {
                    pos_ += 4;
                    out = HavenJson();
                    return true;
                }
                return false;
            }
            if (c == '-' || (c >= '0' && c <= '9')) {
                return parse_number(out);
            }
            return false;
        }
    };
};

} // namespace haven
