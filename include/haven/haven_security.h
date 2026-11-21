#pragma once

#include <string>
#include <vector>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <chrono>

namespace haven {

// Sovereign Haven Cluster Authentication & Security Protocol (SHCP)
class HavenSecurity {
public:
    // Generates a cryptographically secure 256-bit (32-byte) hex cluster token
    static std::string generate_cluster_token() {
        std::random_device rd;
        // Mix with high-resolution clock entropy
        auto seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::mt19937_64 gen(rd() ^ seed);
        std::uniform_int_distribution<uint64_t> dist;

        std::ostringstream ss;
        ss << std::hex << std::setfill('0');
        for (int i = 0; i < 4; ++i) {
            uint64_t part = (dist(gen) ^ (static_cast<uint64_t>(rd()) << 32) ^ rd());
            ss << std::setw(16) << part;
        }
        return ss.str();
    }

    // Encodes a human-friendly cluster pairing URI
    static std::string create_invite_uri(const std::string& host, uint16_t port, 
                                         const std::string& token, const std::string& name = "HavenCluster") {
        std::ostringstream ss;
        ss << "haven://cluster?host=" << host 
           << "&port=" << port 
           << "&token=" << token 
           << "&name=" << name;
        return ss.str();
    }

    // Parses a cluster pairing URI or raw token
    static bool parse_invite_uri(const std::string& uri, std::string& out_host, uint16_t& out_port,
                                 std::string& out_token, std::string& out_name) {
        if (uri.rfind("haven://cluster?", 0) == 0) {
            std::string query = uri.substr(16);
            std::istringstream qss(query);
            std::string param;
            while (std::getline(qss, param, '&')) {
                auto eq = param.find('=');
                if (eq != std::string::npos) {
                    std::string key = param.substr(0, eq);
                    std::string val = param.substr(eq + 1);
                    if (key == "host") out_host = val;
                    else if (key == "port") out_port = static_cast<uint16_t>(std::stoi(val));
                    else if (key == "token") out_token = val;
                    else if (key == "name") out_name = val;
                }
            }
            return !out_token.empty();
        }

        // Check if raw token provided
        if (uri.length() == 64) {
            out_token = uri;
            return true;
        }

        return false;
    }

    // Constant-time token verification to prevent timing side-channel attacks
    static bool constant_time_verify(const std::string& a, const std::string& b) {
        if (a.empty() || b.empty() || a.length() != b.length()) {
            return false;
        }

        volatile unsigned char result = 0;
        for (size_t i = 0; i < a.length(); ++i) {
            result |= static_cast<unsigned char>(a[i] ^ b[i]);
        }
        return result == 0;
    }

    // Simple stream cipher masking for in-flight activation vectors using token seed
    static void mask_activation_payload(float* data, size_t count, const std::string& token, uint64_t seq_id, uint32_t step) {
        if (token.empty() || data == nullptr || count == 0) return;

        uint64_t key_seed = 0;
        for (size_t i = 0; i < token.length(); ++i) {
            key_seed = (key_seed * 131ULL) + static_cast<uint64_t>(token[i]);
        }
        key_seed ^= (seq_id * 2654435761ULL) ^ (static_cast<uint64_t>(step) * 11400714819323198485ULL);

        // PRNG mask generator
        uint64_t state = key_seed;
        for (size_t i = 0; i < count; ++i) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            // XOR raw bits of the IEEE 754 float with deterministic keystream
            uint32_t* raw_bits = reinterpret_cast<uint32_t*>(&data[i]);
            *raw_bits ^= static_cast<uint32_t>(state >> 32);
        }
    }
};

} // namespace haven
