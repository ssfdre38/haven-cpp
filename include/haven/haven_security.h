#pragma once

#include <string>
#include <vector>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <charconv>
#include <fstream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wincrypt.h>
#endif

namespace haven {

// Sovereign Haven Cluster Authentication & Security Protocol (SHCP)
class HavenSecurity {
public:
    // Generates a cryptographically secure 256-bit (32-byte) hex cluster token
    static std::string generate_cluster_token() {
        uint8_t bytes[32] = {0};
        bool filled = false;
#ifdef _WIN32
        HCRYPTPROV hProv = 0;
        if (CryptAcquireContextA(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT | CRYPT_SILENT)) {
            if (CryptGenRandom(hProv, sizeof(bytes), bytes)) {
                filled = true;
            }
            CryptReleaseContext(hProv, 0);
        }
#else
        std::ifstream urandom("/dev/urandom", std::ios::binary);
        if (urandom.is_open()) {
            urandom.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
            if (urandom.gcount() == static_cast<std::streamsize>(sizeof(bytes))) {
                filled = true;
            }
        }
#endif
        if (!filled) {
            std::random_device rd;
            for (size_t i = 0; i < sizeof(bytes); i += 4) {
                uint32_t v = rd();
                size_t n = std::min<size_t>(4, sizeof(bytes) - i);
                std::memcpy(bytes + i, &v, n);
            }
        }

        std::ostringstream ss;
        ss << std::hex << std::setfill('0');
        for (size_t i = 0; i < sizeof(bytes); ++i) {
            ss << std::setw(2) << static_cast<int>(bytes[i]);
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
                    else if (key == "port") {
                        uint32_t p = 0;
                        auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), p);
                        if (ec == std::errc() && ptr == val.data() + val.size() && p > 0 && p <= 65535) {
                            out_port = static_cast<uint16_t>(p);
                        }
                    }
                    else if (key == "token") out_token = val;
                    else if (key == "name") out_name = val;
                }
            }
            return !out_token.empty();
        }

        // Check if raw 256-bit hex token provided
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
            result = static_cast<unsigned char>(result | static_cast<unsigned char>(a[i] ^ b[i]));
        }
        return result == 0;
    }

    // RFC 8439 ChaCha20 Stream Cipher masking for in-flight activation vectors
    // Uses 256-bit key derived from cluster token and 96-bit nonce from (seq_id, step)
    static void mask_activation_payload(float* data, size_t count, const std::string& token, uint64_t seq_id, uint32_t step) {
        if (token.empty() || data == nullptr || count == 0) return;

        // Derive 256-bit (32-byte) key from token
        uint8_t key_bytes[32] = {0};
        bool hex_ok = (token.size() == 64);
        if (hex_ok) {
            for (size_t i = 0; i < 32; ++i) {
                auto hex_nibble = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                    return -1;
                };
                int hi = hex_nibble(token[i * 2]);
                int lo = hex_nibble(token[i * 2 + 1]);
                if (hi < 0 || lo < 0) {
                    hex_ok = false;
                    break;
                }
                key_bytes[i] = static_cast<uint8_t>((hi << 4) | lo);
            }
        }
        if (!hex_ok) {
            // Mix arbitrary-length token into 32 bytes via non-linear SplitMix rounds
            std::memset(key_bytes, 0, sizeof(key_bytes));
            uint64_t h[4] = {
                0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
                0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL
            };
            for (size_t i = 0; i < token.size(); ++i) {
                uint64_t lane = h[i & 3] ^ (static_cast<uint64_t>(static_cast<uint8_t>(token[i])) + 0x9e3779b97f4a7c15ULL + (i * 0xbf58476d1ce4e5b9ULL));
                lane = (lane ^ (lane >> 30)) * 0xbf58476d1ce4e5b9ULL;
                lane = (lane ^ (lane >> 27)) * 0x94d049bb133111ebULL;
                h[i & 3] = lane ^ (lane >> 31);
            }
            for (int r = 0; r < 4; ++r) {
                h[r] ^= h[(r + 1) & 3] * 0x9e3779b97f4a7c15ULL;
                h[r] = (h[r] ^ (h[r] >> 30)) * 0xbf58476d1ce4e5b9ULL;
            }
            std::memcpy(key_bytes, h, 32);
        }

        auto load_le32 = [](const uint8_t* p) -> uint32_t {
            return static_cast<uint32_t>(p[0]) |
                   (static_cast<uint32_t>(p[1]) << 8) |
                   (static_cast<uint32_t>(p[2]) << 16) |
                   (static_cast<uint32_t>(p[3]) << 24);
        };

        auto rotl32 = [](uint32_t v, int c) -> uint32_t {
            return (v << c) | (v >> (32 - c));
        };

        auto qr = [&](uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
            a += b; d ^= a; d = rotl32(d, 16);
            c += d; b ^= c; b = rotl32(b, 12);
            a += b; d ^= a; d = rotl32(d, 8);
            c += d; b ^= c; b = rotl32(b, 7);
        };

        uint32_t init_state[16];
        // ChaCha20 "expand 32-byte k" constants
        init_state[0] = 0x61707865U;
        init_state[1] = 0x3320646eU;
        init_state[2] = 0x79622d32U;
        init_state[3] = 0x6b206574U;
        for (int i = 0; i < 8; ++i) {
            init_state[4 + i] = load_le32(key_bytes + i * 4);
        }
        init_state[12] = 0; // Block counter
        init_state[13] = static_cast<uint32_t>(seq_id & 0xFFFFFFFFULL);
        init_state[14] = static_cast<uint32_t>(seq_id >> 32);
        init_state[15] = step;

        uint8_t* raw = reinterpret_cast<uint8_t*>(data);
        const size_t total_bytes = count * sizeof(float);
        size_t offset = 0;
        uint32_t counter = 0;

        while (offset < total_bytes) {
            init_state[12] = counter++;
            uint32_t x[16];
            std::memcpy(x, init_state, sizeof(x));

            for (int r = 0; r < 10; ++r) {
                // Column rounds
                qr(x[0], x[4], x[8],  x[12]);
                qr(x[1], x[5], x[9],  x[13]);
                qr(x[2], x[6], x[10], x[14]);
                qr(x[3], x[7], x[11], x[15]);
                // Diagonal rounds
                qr(x[0], x[5], x[10], x[15]);
                qr(x[1], x[6], x[11], x[12]);
                qr(x[2], x[7], x[8],  x[13]);
                qr(x[3], x[4], x[9],  x[14]);
            }

            uint8_t keystream[64];
            for (int i = 0; i < 16; ++i) {
                uint32_t w = x[i] + init_state[i];
                keystream[i * 4 + 0] = static_cast<uint8_t>(w & 0xFF);
                keystream[i * 4 + 1] = static_cast<uint8_t>((w >> 8) & 0xFF);
                keystream[i * 4 + 2] = static_cast<uint8_t>((w >> 16) & 0xFF);
                keystream[i * 4 + 3] = static_cast<uint8_t>((w >> 24) & 0xFF);
            }

            size_t chunk = std::min<size_t>(64, total_bytes - offset);
            for (size_t i = 0; i < chunk; ++i) {
                raw[offset + i] ^= keystream[i];
            }
            offset += chunk;
        }
    }
};

} // namespace haven
