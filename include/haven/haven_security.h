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

// Sovereign Haven Cluster Authentication & Security Protocol (SHCP v2)
// Cryptographic Primitives:
// - OS CSPRNG (CryptGenRandom / /dev/urandom)
// - FIPS 180-4 SHA-256 & RFC 2104 HMAC-SHA256 (zero-secret-on-wire mutual challenge-response)
// - RFC 8439 ChaCha20-Poly1305 AEAD (authenticated header + activation tensor encryption)
class HavenSecurity {
public:
    // Securely zeroes sensitive key material to resist compiler dead-store elimination
    static void secure_zero(void* ptr, size_t len) {
        if (!ptr || len == 0) return;
        volatile uint8_t* p = static_cast<volatile uint8_t*>(ptr);
        while (len--) *p++ = 0;
    }

    // Fills buffer with OS-backed cryptographically secure random bytes
    static bool generate_random_bytes(uint8_t* out, size_t len) {
        if (!out || len == 0) return false;
        bool filled = false;
#ifdef _WIN32
        HCRYPTPROV hProv = 0;
        if (CryptAcquireContextA(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT | CRYPT_SILENT)) {
            if (CryptGenRandom(hProv, static_cast<DWORD>(len), out)) {
                filled = true;
            }
            CryptReleaseContext(hProv, 0);
        }
#else
        std::ifstream urandom("/dev/urandom", std::ios::binary);
        if (urandom.is_open()) {
            urandom.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(len));
            if (urandom.gcount() == static_cast<std::streamsize>(len)) {
                filled = true;
            }
        }
#endif
        if (!filled) {
            std::random_device rd;
            for (size_t i = 0; i < len; i += 4) {
                uint32_t v = rd();
                size_t n = std::min<size_t>(4, len - i);
                std::memcpy(out + i, &v, n);
            }
        }
        return true;
    }

    // Generates a cryptographically secure 256-bit (32-byte) hex cluster token
    static std::string generate_cluster_token() {
        uint8_t bytes[32] = {0};
        generate_random_bytes(bytes, sizeof(bytes));

        std::ostringstream ss;
        ss << std::hex << std::setfill('0');
        for (size_t i = 0; i < sizeof(bytes); ++i) {
            ss << std::setw(2) << static_cast<int>(bytes[i]);
        }
        secure_zero(bytes, sizeof(bytes));
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

        if (uri.length() == 64) {
            out_token = uri;
            return true;
        }

        return false;
    }

    // Constant-time byte comparison
    static bool constant_time_memeq(const void* a, const void* b, size_t len) {
        if (!a || !b || len == 0) return false;
        const uint8_t* pa = static_cast<const uint8_t*>(a);
        const uint8_t* pb = static_cast<const uint8_t*>(b);
        volatile uint8_t diff = 0;
        for (size_t i = 0; i < len; ++i) {
            diff = static_cast<uint8_t>(diff | static_cast<uint8_t>(pa[i] ^ pb[i]));
        }
        return diff == 0;
    }

    // Constant-time token verification to prevent timing side-channel attacks
    static bool constant_time_verify(const std::string& a, const std::string& b) {
        if (a.empty() || b.empty() || a.length() != b.length()) {
            return false;
        }
        return constant_time_memeq(a.data(), b.data(), a.length());
    }

    // =========================================================================
    // FIPS 180-4 SHA-256
    // =========================================================================
    static void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
        static constexpr uint32_t K[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
        };

        auto rotr = [](uint32_t x, int n) -> uint32_t { return (x >> n) | (x << (32 - n)); };
        uint32_t h[8] = {
            0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
            0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
        };

        auto compress = [&](const uint8_t block[64]) {
            uint32_t w[64];
            for (int i = 0; i < 16; ++i) {
                w[i] = (static_cast<uint32_t>(block[i * 4 + 0]) << 24) |
                       (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
                       (static_cast<uint32_t>(block[i * 4 + 2]) << 8)  |
                       (static_cast<uint32_t>(block[i * 4 + 3]));
            }
            for (int i = 16; i < 64; ++i) {
                uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }
            uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
            uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
            for (int i = 0; i < 64; ++i) {
                uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
                uint32_t ch = (e & f) ^ ((~e) & g);
                uint32_t temp1 = hh + S1 + ch + K[i] + w[i];
                uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
                uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                uint32_t temp2 = S0 + maj;
                hh = g; g = f; f = e; e = d + temp1;
                d = c; c = b; b = a; a = temp1 + temp2;
            }
            h[0] += a; h[1] += b; h[2] += c; h[3] += d;
            h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
        };

        size_t offset = 0;
        while (offset + 64 <= len) {
            compress(data + offset);
            offset += 64;
        }

        uint8_t tail[128] = {0};
        size_t rem = len - offset;
        if (rem > 0 && data) {
            std::memcpy(tail, data + offset, rem);
        }
        tail[rem] = 0x80;
        size_t padded_len = (rem < 56) ? 64 : 128;
        uint64_t bit_len = static_cast<uint64_t>(len) * 8ULL;
        for (int i = 0; i < 8; ++i) {
            tail[padded_len - 1 - i] = static_cast<uint8_t>((bit_len >> (i * 8)) & 0xFFU);
        }
        compress(tail);
        if (padded_len == 128) {
            compress(tail + 64);
        }

        for (int i = 0; i < 8; ++i) {
            out[i * 4 + 0] = static_cast<uint8_t>((h[i] >> 24) & 0xFFU);
            out[i * 4 + 1] = static_cast<uint8_t>((h[i] >> 16) & 0xFFU);
            out[i * 4 + 2] = static_cast<uint8_t>((h[i] >> 8) & 0xFFU);
            out[i * 4 + 3] = static_cast<uint8_t>(h[i] & 0xFFU);
        }
        secure_zero(tail, sizeof(tail));
    }

    // =========================================================================
    // RFC 2104 HMAC-SHA256
    // =========================================================================
    static void hmac_sha256(const uint8_t* key, size_t key_len,
                            const uint8_t* msg, size_t msg_len,
                            uint8_t out[32]) {
        uint8_t k_block[64] = {0};
        if (key && key_len > 64) {
            sha256(key, key_len, k_block);
        } else if (key && key_len > 0) {
            std::memcpy(k_block, key, key_len);
        }

        std::vector<uint8_t> inner(64 + msg_len);
        for (int i = 0; i < 64; ++i) inner[i] = static_cast<uint8_t>(k_block[i] ^ 0x36U);
        if (msg && msg_len > 0) std::memcpy(inner.data() + 64, msg, msg_len);

        uint8_t inner_hash[32];
        sha256(inner.data(), inner.size(), inner_hash);

        uint8_t outer[64 + 32];
        for (int i = 0; i < 64; ++i) outer[i] = static_cast<uint8_t>(k_block[i] ^ 0x5CU);
        std::memcpy(outer + 64, inner_hash, 32);

        sha256(outer, sizeof(outer), out);
        secure_zero(k_block, sizeof(k_block));
        secure_zero(inner_hash, sizeof(inner_hash));
        secure_zero(outer, sizeof(outer));
        if (!inner.empty()) secure_zero(inner.data(), inner.size());
    }

    // =========================================================================
    // RFC 8439 ChaCha20
    // =========================================================================
    static void chacha20_block(const uint8_t key[32], uint32_t counter, const uint8_t nonce[12], uint8_t out[64]) {
        auto load_le32 = [](const uint8_t* p) -> uint32_t {
            return static_cast<uint32_t>(p[0]) |
                   (static_cast<uint32_t>(p[1]) << 8) |
                   (static_cast<uint32_t>(p[2]) << 16) |
                   (static_cast<uint32_t>(p[3]) << 24);
        };
        auto rotl32 = [](uint32_t v, int c) -> uint32_t { return (v << c) | (v >> (32 - c)); };
        auto qr = [&](uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
            a += b; d ^= a; d = rotl32(d, 16);
            c += d; b ^= c; b = rotl32(b, 12);
            a += b; d ^= a; d = rotl32(d, 8);
            c += d; b ^= c; b = rotl32(b, 7);
        };

        uint32_t state[16] = {
            0x61707865U, 0x3320646eU, 0x79622d32U, 0x6b206574U,
            load_le32(key + 0),  load_le32(key + 4),  load_le32(key + 8),  load_le32(key + 12),
            load_le32(key + 16), load_le32(key + 20), load_le32(key + 24), load_le32(key + 28),
            counter,
            load_le32(nonce + 0), load_le32(nonce + 4), load_le32(nonce + 8)
        };

        uint32_t x[16];
        std::memcpy(x, state, sizeof(x));
        for (int r = 0; r < 10; ++r) {
            qr(x[0], x[4], x[8],  x[12]);
            qr(x[1], x[5], x[9],  x[13]);
            qr(x[2], x[6], x[10], x[14]);
            qr(x[3], x[7], x[11], x[15]);
            qr(x[0], x[5], x[10], x[15]);
            qr(x[1], x[6], x[11], x[12]);
            qr(x[2], x[7], x[8],  x[13]);
            qr(x[3], x[4], x[9],  x[14]);
        }

        for (int i = 0; i < 16; ++i) {
            uint32_t w = x[i] + state[i];
            out[i * 4 + 0] = static_cast<uint8_t>(w & 0xFFU);
            out[i * 4 + 1] = static_cast<uint8_t>((w >> 8) & 0xFFU);
            out[i * 4 + 2] = static_cast<uint8_t>((w >> 16) & 0xFFU);
            out[i * 4 + 3] = static_cast<uint8_t>((w >> 24) & 0xFFU);
        }
    }

    static void chacha20_xor(const uint8_t key[32], uint32_t init_counter, const uint8_t nonce[12],
                             const uint8_t* in, uint8_t* out, size_t len) {
        size_t offset = 0;
        uint32_t counter = init_counter;
        uint8_t ks[64];
        while (offset < len) {
            chacha20_block(key, counter++, nonce, ks);
            size_t chunk = std::min<size_t>(64, len - offset);
            for (size_t i = 0; i < chunk; ++i) {
                out[offset + i] = static_cast<uint8_t>(in[offset + i] ^ ks[i]);
            }
            offset += chunk;
        }
        secure_zero(ks, sizeof(ks));
    }

    // =========================================================================
    // RFC 8439 Poly1305 MAC (portable 5x26-bit constant-time limbs)
    // =========================================================================
    static void poly1305_mac(const uint8_t key[32], const uint8_t* msg, size_t len, uint8_t tag[16]) {
        auto load_le32 = [](const uint8_t* p) -> uint32_t {
            return static_cast<uint32_t>(p[0]) |
                   (static_cast<uint32_t>(p[1]) << 8) |
                   (static_cast<uint32_t>(p[2]) << 16) |
                   (static_cast<uint32_t>(p[3]) << 24);
        };

        uint8_t rb[16];
        std::memcpy(rb, key, 16);
        rb[3]  &= 15; rb[7]  &= 15; rb[11] &= 15; rb[15] &= 15;
        rb[4]  &= 252; rb[8]  &= 252; rb[12] &= 252;

        uint32_t r0 = (load_le32(rb + 0)) & 0x3ffffffU;
        uint32_t r1 = (load_le32(rb + 3) >> 2) & 0x3ffffffU;
        uint32_t r2 = (load_le32(rb + 6) >> 4) & 0x3ffffffU;
        uint32_t r3 = (load_le32(rb + 9) >> 6) & 0x3ffffffU;
        uint32_t r4 = (load_le32(rb + 12) >> 8) & 0x3ffffffU;

        uint32_t s1 = r1 * 5U;
        uint32_t s2 = r2 * 5U;
        uint32_t s3 = r3 * 5U;
        uint32_t s4 = r4 * 5U;

        uint32_t pad0 = load_le32(key + 16);
        uint32_t pad1 = load_le32(key + 20);
        uint32_t pad2 = load_le32(key + 24);
        uint32_t pad3 = load_le32(key + 28);

        uint32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0, h4 = 0;

        size_t offset = 0;
        while (offset < len) {
            uint8_t blk[16] = {0};
            size_t rem = len - offset;
            uint32_t hibit = 0;
            if (rem >= 16) {
                std::memcpy(blk, msg + offset, 16);
                hibit = (1U << 24);
                offset += 16;
            } else {
                std::memcpy(blk, msg + offset, rem);
                blk[rem] = 1;
                hibit = 0;
                offset += rem;
            }

            h0 += (load_le32(blk + 0)) & 0x3ffffffU;
            h1 += (load_le32(blk + 3) >> 2) & 0x3ffffffU;
            h2 += (load_le32(blk + 6) >> 4) & 0x3ffffffU;
            h3 += (load_le32(blk + 9) >> 6) & 0x3ffffffU;
            h4 += (load_le32(blk + 12) >> 8) | hibit;

            uint64_t d0 = static_cast<uint64_t>(h0) * r0 + static_cast<uint64_t>(h1) * s4 +
                          static_cast<uint64_t>(h2) * s3 + static_cast<uint64_t>(h3) * s2 +
                          static_cast<uint64_t>(h4) * s1;
            uint64_t d1 = static_cast<uint64_t>(h0) * r1 + static_cast<uint64_t>(h1) * r0 +
                          static_cast<uint64_t>(h2) * s4 + static_cast<uint64_t>(h3) * s3 +
                          static_cast<uint64_t>(h4) * s2;
            uint64_t d2 = static_cast<uint64_t>(h0) * r2 + static_cast<uint64_t>(h1) * r1 +
                          static_cast<uint64_t>(h2) * r0 + static_cast<uint64_t>(h3) * s4 +
                          static_cast<uint64_t>(h4) * s3;
            uint64_t d3 = static_cast<uint64_t>(h0) * r3 + static_cast<uint64_t>(h1) * r2 +
                          static_cast<uint64_t>(h2) * r1 + static_cast<uint64_t>(h3) * r0 +
                          static_cast<uint64_t>(h4) * s4;
            uint64_t d4 = static_cast<uint64_t>(h0) * r4 + static_cast<uint64_t>(h1) * r3 +
                          static_cast<uint64_t>(h2) * r2 + static_cast<uint64_t>(h3) * r1 +
                          static_cast<uint64_t>(h4) * r0;

            uint32_t c;
            c = static_cast<uint32_t>(d0 >> 26); h0 = static_cast<uint32_t>(d0) & 0x3ffffffU;
            d1 += c; c = static_cast<uint32_t>(d1 >> 26); h1 = static_cast<uint32_t>(d1) & 0x3ffffffU;
            d2 += c; c = static_cast<uint32_t>(d2 >> 26); h2 = static_cast<uint32_t>(d2) & 0x3ffffffU;
            d3 += c; c = static_cast<uint32_t>(d3 >> 26); h3 = static_cast<uint32_t>(d3) & 0x3ffffffU;
            d4 += c; c = static_cast<uint32_t>(d4 >> 26); h4 = static_cast<uint32_t>(d4) & 0x3ffffffU;
            h0 += c * 5U; c = (h0 >> 26); h0 &= 0x3ffffffU;
            h1 += c;
        }

        // Full carry propagation
        uint32_t c;
        c = h1 >> 26; h1 &= 0x3ffffffU;
        h2 += c; c = h2 >> 26; h2 &= 0x3ffffffU;
        h3 += c; c = h3 >> 26; h3 &= 0x3ffffffU;
        h4 += c; c = h4 >> 26; h4 &= 0x3ffffffU;
        h0 += c * 5U; c = h0 >> 26; h0 &= 0x3ffffffU;
        h1 += c;

        // Compute h - (2^130 - 5)
        uint32_t g0 = h0 + 5U; c = g0 >> 26; g0 &= 0x3ffffffU;
        uint32_t g1 = h1 + c;  c = g1 >> 26; g1 &= 0x3ffffffU;
        uint32_t g2 = h2 + c;  c = g2 >> 26; g2 &= 0x3ffffffU;
        uint32_t g3 = h3 + c;  c = g3 >> 26; g3 &= 0x3ffffffU;
        uint32_t g4 = h4 + c - (1U << 26);

        // Constant-time select h if h < p else g
        uint32_t mask = (g4 >> 31) - 1U;
        g0 &= mask; g1 &= mask; g2 &= mask; g3 &= mask; g4 &= mask;
        mask = ~mask;
        h0 = (h0 & mask) | g0;
        h1 = (h1 & mask) | g1;
        h2 = (h2 & mask) | g2;
        h3 = (h3 & mask) | g3;
        h4 = (h4 & mask) | g4;

        // Pack into 128-bit result and add s (pad) mod 2^128
        h0 = (h0)       | (h1 << 26);
        h1 = (h1 >> 6)  | (h2 << 20);
        h2 = (h2 >> 12) | (h3 << 14);
        h3 = (h3 >> 18) | (h4 << 8);

        uint64_t f;
        f = static_cast<uint64_t>(h0) + pad0;               h0 = static_cast<uint32_t>(f);
        f = static_cast<uint64_t>(h1) + pad1 + (f >> 32);   h1 = static_cast<uint32_t>(f);
        f = static_cast<uint64_t>(h2) + pad2 + (f >> 32);   h2 = static_cast<uint32_t>(f);
        f = static_cast<uint64_t>(h3) + pad3 + (f >> 32);   h3 = static_cast<uint32_t>(f);

        auto store_le32 = [](uint8_t* p, uint32_t v) {
            p[0] = static_cast<uint8_t>(v & 0xFFU);
            p[1] = static_cast<uint8_t>((v >> 8) & 0xFFU);
            p[2] = static_cast<uint8_t>((v >> 16) & 0xFFU);
            p[3] = static_cast<uint8_t>((v >> 24) & 0xFFU);
        };
        store_le32(tag + 0,  h0);
        store_le32(tag + 4,  h1);
        store_le32(tag + 8,  h2);
        store_le32(tag + 12, h3);
        secure_zero(rb, sizeof(rb));
    }

    // =========================================================================
    // RFC 8439 ChaCha20-Poly1305 AEAD
    // =========================================================================
    static void build_aead_mac_input(const uint8_t* aad, size_t aad_len,
                                     const uint8_t* ct, size_t ct_len,
                                     std::vector<uint8_t>& mac_data) {
        size_t aad_pad = (16 - (aad_len % 16)) % 16;
        size_t ct_pad  = (16 - (ct_len % 16)) % 16;
        mac_data.clear();
        mac_data.reserve(aad_len + aad_pad + ct_len + ct_pad + 16);
        if (aad && aad_len > 0) mac_data.insert(mac_data.end(), aad, aad + aad_len);
        mac_data.insert(mac_data.end(), aad_pad, 0);
        if (ct && ct_len > 0) mac_data.insert(mac_data.end(), ct, ct + ct_len);
        mac_data.insert(mac_data.end(), ct_pad, 0);
        auto append_le64 = [&](uint64_t v) {
            for (int i = 0; i < 8; ++i) mac_data.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFFU));
        };
        append_le64(static_cast<uint64_t>(aad_len));
        append_le64(static_cast<uint64_t>(ct_len));
    }

    static void aead_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t* aad, size_t aad_len,
                             const uint8_t* plaintext, size_t pt_len,
                             uint8_t* ciphertext, uint8_t tag[16]) {
        uint8_t poly_key_block[64];
        chacha20_block(key, 0, nonce, poly_key_block);

        if (pt_len > 0 && plaintext && ciphertext) {
            chacha20_xor(key, 1, nonce, plaintext, ciphertext, pt_len);
        }

        std::vector<uint8_t> mac_data;
        build_aead_mac_input(aad, aad_len, ciphertext, pt_len, mac_data);
        poly1305_mac(poly_key_block, mac_data.data(), mac_data.size(), tag);
        secure_zero(poly_key_block, sizeof(poly_key_block));
    }

    static bool aead_decrypt(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t* aad, size_t aad_len,
                             const uint8_t* ciphertext, size_t ct_len,
                             const uint8_t expected_tag[16],
                             uint8_t* plaintext) {
        uint8_t poly_key_block[64];
        chacha20_block(key, 0, nonce, poly_key_block);

        std::vector<uint8_t> mac_data;
        build_aead_mac_input(aad, aad_len, ciphertext, ct_len, mac_data);
        uint8_t computed_tag[16];
        poly1305_mac(poly_key_block, mac_data.data(), mac_data.size(), computed_tag);
        secure_zero(poly_key_block, sizeof(poly_key_block));

        bool valid = constant_time_memeq(computed_tag, expected_tag, 16);
        secure_zero(computed_tag, sizeof(computed_tag));
        if (!valid) {
            return false;
        }

        if (ct_len > 0 && ciphertext && plaintext) {
            chacha20_xor(key, 1, nonce, ciphertext, plaintext, ct_len);
        }
        return true;
    }

    // =========================================================================
    // SHCP v2 Mutual Challenge-Response & Per-Session Key Derivation
    // =========================================================================
    static void derive_master_key(const std::string& token, uint8_t out_key[32]) {
        std::string labeled = "haven-shcp-master-v2:" + token;
        sha256(reinterpret_cast<const uint8_t*>(labeled.data()), labeled.size(), out_key);
    }

    static void derive_shcp_proof(const std::string& token, const char* label,
                                  const uint8_t server_nonce[32], const uint8_t client_nonce[32],
                                  uint16_t node_id, uint8_t out_proof[32]) {
        uint8_t master_key[32];
        derive_master_key(token, master_key);
        size_t lbl_len = std::strlen(label);
        std::vector<uint8_t> msg(lbl_len + 66);
        std::memcpy(msg.data(), label, lbl_len);
        std::memcpy(msg.data() + lbl_len, server_nonce, 32);
        std::memcpy(msg.data() + lbl_len + 32, client_nonce, 32);
        msg[lbl_len + 64] = static_cast<uint8_t>(node_id & 0xFFU);
        msg[lbl_len + 65] = static_cast<uint8_t>((node_id >> 8) & 0xFFU);
        hmac_sha256(master_key, 32, msg.data(), msg.size(), out_proof);
        secure_zero(master_key, sizeof(master_key));
    }

    static void derive_shcp_client_proof(const std::string& token,
                                         const uint8_t server_nonce[32], const uint8_t client_nonce[32],
                                         uint16_t node_id, uint8_t out_proof[32]) {
        derive_shcp_proof(token, "haven-shcp-client-v2", server_nonce, client_nonce, node_id, out_proof);
    }

    static void derive_shcp_server_proof(const std::string& token,
                                         const uint8_t server_nonce[32], const uint8_t client_nonce[32],
                                         uint16_t node_id, uint8_t out_proof[32]) {
        derive_shcp_proof(token, "haven-shcp-server-v2", server_nonce, client_nonce, node_id, out_proof);
    }

    static void derive_shcp_session_key(const std::string& token,
                                        const uint8_t server_nonce[32], const uint8_t client_nonce[32],
                                        uint8_t out_key[32]) {
        uint8_t master_key[32];
        derive_master_key(token, master_key);
        const char* lbl = "haven-shcp-session-v2";
        size_t lbl_len = std::strlen(lbl);
        std::vector<uint8_t> msg(lbl_len + 64);
        std::memcpy(msg.data(), lbl, lbl_len);
        std::memcpy(msg.data() + lbl_len, server_nonce, 32);
        std::memcpy(msg.data() + lbl_len + 32, client_nonce, 32);
        hmac_sha256(master_key, 32, msg.data(), msg.size(), out_key);
        secure_zero(master_key, sizeof(master_key));
    }

    // Legacy / standalone ChaCha20 masking helper (retained for compatibility)
    static void mask_activation_payload(float* data, size_t count, const std::string& token, uint64_t seq_id, uint32_t step) {
        if (token.empty() || data == nullptr || count == 0) return;
        uint8_t key_bytes[32];
        derive_master_key(token, key_bytes);
        uint8_t nonce[12];
        for (int i = 0; i < 8; ++i) nonce[i] = static_cast<uint8_t>((seq_id >> (i * 8)) & 0xFFU);
        for (int i = 0; i < 4; ++i) nonce[8 + i] = static_cast<uint8_t>((step >> (i * 8)) & 0xFFU);
        uint8_t* raw = reinterpret_cast<uint8_t*>(data);
        chacha20_xor(key_bytes, 0, nonce, raw, raw, count * sizeof(float));
        secure_zero(key_bytes, sizeof(key_bytes));
    }
};

} // namespace haven
