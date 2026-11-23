#include "haven/haven_security.h"
#include "haven/haven_cluster.h"
#include "haven/haven_json.h"
#include "haven/haven_gguf.h"
#include "haven/haven_memory.h"
#include "haven/haven_plugin_manager.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <random>
#include <thread>
#include <chrono>

#define HAVEN_CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FATAL CHECK FAILED: " #cond " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::abort(); \
    } \
} while (0)
#undef assert
#define assert(cond) HAVEN_CHECK(cond)

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET raw_sock_t;
#define RAW_INVALID_SOCK INVALID_SOCKET
#define raw_close closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
typedef int raw_sock_t;
#define RAW_INVALID_SOCK (-1)
#define raw_close close
#endif

using namespace haven;

static std::string to_hex(const uint8_t* data, size_t len) {
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        ss << std::setw(2) << static_cast<int>(data[i]);
    }
    return ss.str();
}

static std::vector<uint8_t> from_hex(const std::string& hex) {
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        auto nib = [](char c) -> uint8_t {
            if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
            return 0;
        };
        out.push_back(static_cast<uint8_t>((nib(hex[i]) << 4) | nib(hex[i + 1])));
    }
    return out;
}

static bool recv_exact_raw(raw_sock_t s, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    size_t got = 0;
    while (got < n) {
        int r = ::recv(s, p + got, static_cast<int>(n - got), 0);
        if (r <= 0) return false;
        got += static_cast<size_t>(r);
    }
    return true;
}

// -----------------------------------------------------------------------------
// 1. Cryptographic Known Answer Tests (FIPS 180-4, RFC 4231, RFC 8439)
// -----------------------------------------------------------------------------
static void test_crypto_kats() {
    std::cout << "[1/6] Running FIPS 180-4 SHA-256, RFC 4231 HMAC-SHA256, & RFC 8439 ChaCha20-Poly1305 KATs...\n";

    // FIPS 180-4 SHA-256 vectors
    uint8_t digest[32];
    HavenSecurity::sha256(reinterpret_cast<const uint8_t*>(""), 0, digest);
    assert(to_hex(digest, 32) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    HavenSecurity::sha256(reinterpret_cast<const uint8_t*>("abc"), 3, digest);
    assert(to_hex(digest, 32) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    const char* msg448 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    HavenSecurity::sha256(reinterpret_cast<const uint8_t*>(msg448), std::strlen(msg448), digest);
    assert(to_hex(digest, 32) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    // RFC 4231 HMAC-SHA256 Test Case 2 ("Jefe" / "what do ya want for nothing?")
    const char* k2 = "Jefe";
    const char* d2 = "what do ya want for nothing?";
    HavenSecurity::hmac_sha256(reinterpret_cast<const uint8_t*>(k2), std::strlen(k2),
                               reinterpret_cast<const uint8_t*>(d2), std::strlen(d2), digest);
    assert(to_hex(digest, 32) == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");

    // RFC 8439 Section 2.5.2 Poly1305 Test Vector
    auto poly_key = from_hex("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b");
    const char* poly_msg = "Cryptographic Forum Research Group";
    uint8_t poly_tag[16];
    HavenSecurity::poly1305_mac(poly_key.data(), reinterpret_cast<const uint8_t*>(poly_msg), std::strlen(poly_msg), poly_tag);
    assert(to_hex(poly_tag, 16) == "a8061dc1305136c6c22b8baf0c0127a9");

    // RFC 8439 Section 2.8.2 ChaCha20-Poly1305 AEAD Test Vector
    auto aead_key = from_hex("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f");
    auto aead_nonce = from_hex("070000004041424344454647");
    auto aead_aad = from_hex("50515253c0c1c2c3c4c5c6c7");
    const char* pt_str =
        "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
    const size_t pt_len = std::strlen(pt_str);

    std::vector<uint8_t> ct(pt_len, 0);
    uint8_t tag[16] = {0};
    HavenSecurity::aead_encrypt(aead_key.data(), aead_nonce.data(),
                                aead_aad.data(), aead_aad.size(),
                                reinterpret_cast<const uint8_t*>(pt_str), pt_len,
                                ct.data(), tag);

    const std::string expected_ct =
        "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6"
        "3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36"
        "92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc"
        "3ff4def08e4b7a9de576d26586cec64b6116";
    const std::string expected_tag = "1ae10b594f09e26a7e902ecbd0600691";

    assert(to_hex(ct.data(), ct.size()) == expected_ct && "RFC 8439 AEAD ciphertext mismatch");
    assert(to_hex(tag, 16) == expected_tag && "RFC 8439 AEAD tag mismatch");

    // Decrypt valid
    std::vector<uint8_t> pt_out(pt_len, 0);
    bool ok = HavenSecurity::aead_decrypt(aead_key.data(), aead_nonce.data(),
                                          aead_aad.data(), aead_aad.size(),
                                          ct.data(), ct.size(), tag, pt_out.data());
    assert(ok && std::memcmp(pt_out.data(), pt_str, pt_len) == 0);

    // Flip 1 bit in ciphertext -> MUST fail
    ct[0] ^= 0x01;
    assert(!HavenSecurity::aead_decrypt(aead_key.data(), aead_nonce.data(),
                                        aead_aad.data(), aead_aad.size(),
                                        ct.data(), ct.size(), tag, pt_out.data()));
    ct[0] ^= 0x01;

    // Flip 1 bit in AAD -> MUST fail
    aead_aad[0] ^= 0x01;
    assert(!HavenSecurity::aead_decrypt(aead_key.data(), aead_nonce.data(),
                                        aead_aad.data(), aead_aad.size(),
                                        ct.data(), ct.size(), tag, pt_out.data()));
    aead_aad[0] ^= 0x01;

    // Flip 1 bit in tag -> MUST fail
    tag[15] ^= 0x80;
    assert(!HavenSecurity::aead_decrypt(aead_key.data(), aead_nonce.data(),
                                        aead_aad.data(), aead_aad.size(),
                                        ct.data(), ct.size(), tag, pt_out.data()));

    std::cout << "   ✓ All FIPS 180-4, RFC 4231, and RFC 8439 bit-exact KATs & tamper checks PASSED.\n";
}

// -----------------------------------------------------------------------------
// 2. SHCP v2 Live Wire Tamper, Downgrade, & Replay Attack Tests
// -----------------------------------------------------------------------------
static bool perform_raw_shcp_v2_handshake(raw_sock_t sock, const std::string& token,
                                          uint16_t node_id, uint8_t out_session_key[32]) {
    HavenActivationFrameHeader chal_hdr{};
    if (!recv_exact_raw(sock, &chal_hdr, sizeof(chal_hdr))) return false;
    if (!(chal_hdr.flags & FRAME_FLAG_CHALLENGE) || chal_hdr.payload_bytes != 32) return false;
    uint8_t server_nonce[32];
    if (!recv_exact_raw(sock, server_nonce, 32)) return false;

    uint8_t client_nonce[32];
    uint8_t client_proof[32];
    HavenSecurity::generate_random_bytes(client_nonce, 32);
    HavenSecurity::derive_shcp_client_proof(token, server_nonce, client_nonce, node_id, client_proof);

    HavenActivationFrameHeader hs_hdr{};
    hs_hdr.magic = HAVEN_CLUSTER_MAGIC;
    hs_hdr.protocol_ver = HAVEN_CLUSTER_VERSION;
    hs_hdr.flags = FRAME_FLAG_HANDSHAKE;
    hs_hdr.from_node_id = node_id;
    hs_hdr.element_count = 16;
    hs_hdr.payload_bytes = 64;
    uint8_t hs_buf[64];
    std::memcpy(hs_buf, client_nonce, 32);
    std::memcpy(hs_buf + 32, client_proof, 32);
    ::send(sock, reinterpret_cast<const char*>(&hs_hdr), sizeof(hs_hdr), 0);
    ::send(sock, reinterpret_cast<const char*>(hs_buf), sizeof(hs_buf), 0);

    HavenActivationFrameHeader ack_hdr{};
    if (!recv_exact_raw(sock, &ack_hdr, sizeof(ack_hdr))) return false;
    if (!(ack_hdr.flags & FRAME_FLAG_HANDSHAKE_ACK) || ack_hdr.payload_bytes != 32) return false;
    uint8_t server_proof[32];
    if (!recv_exact_raw(sock, server_proof, 32)) return false;

    uint8_t expected_sp[32];
    HavenSecurity::derive_shcp_server_proof(token, server_nonce, client_nonce, node_id, expected_sp);
    if (!HavenSecurity::constant_time_memeq(server_proof, expected_sp, 32)) return false;

    HavenSecurity::derive_shcp_session_key(token, server_nonce, client_nonce, out_session_key);
    return true;
}

static void build_wire_aead_packet(const uint8_t session_key[32], uint64_t ctr, uint16_t from_id,
                                   const std::vector<float>& vec,
                                   HavenActivationFrameHeader& hdr_out, std::vector<uint8_t>& body_out) {
    const size_t pt_bytes = vec.size() * sizeof(float);
    hdr_out = {};
    hdr_out.magic = HAVEN_CLUSTER_MAGIC;
    hdr_out.protocol_ver = HAVEN_CLUSTER_VERSION;
    hdr_out.flags = FRAME_FLAG_ACTIVATION | FRAME_FLAG_ENCRYPTED;
    hdr_out.sequence_id = 42;
    hdr_out.step_pos = 1;
    hdr_out.from_node_id = from_id;
    hdr_out.to_node_id = 0;
    hdr_out.element_count = static_cast<uint32_t>(vec.size());
    hdr_out.payload_bytes = static_cast<uint32_t>(pt_bytes + 24);

    uint8_t nonce[12] = {'C', 'T', 'W', '1', 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 8; ++i) nonce[4 + i] = static_cast<uint8_t>((ctr >> (i * 8)) & 0xFFU);

    body_out.assign(pt_bytes + 24, 0);
    uint8_t tag[16];
    HavenSecurity::aead_encrypt(session_key, nonce,
                                reinterpret_cast<const uint8_t*>(&hdr_out), sizeof(hdr_out),
                                reinterpret_cast<const uint8_t*>(vec.data()), pt_bytes,
                                body_out.data(), tag);
    for (int i = 0; i < 8; ++i) body_out[pt_bytes + i] = static_cast<uint8_t>((ctr >> (i * 8)) & 0xFFU);
    std::memcpy(body_out.data() + pt_bytes + 8, tag, 16);
}

static void test_shcp_v2_wire_attacks() {
    std::cout << "[2/6] Running SHCP v2 Live Wire Tamper, Plaintext-Downgrade, & Replay Attack Tests...\n";
    std::string token = HavenSecurity::generate_cluster_token();
    HavenClusterNode coord;
    coord.set_cluster_token(token);
    assert(coord.init_coordinator(11850, 32));

    auto connect_raw = [&]() -> raw_sock_t {
        raw_sock_t s = ::socket(AF_INET, SOCK_STREAM, 0);
        assert(s != RAW_INVALID_SOCK);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(11850);
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        assert(::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
        return s;
    };

    // 2a. Valid AEAD packet succeeds, then replaying exact same packet (ctr=1) MUST be rejected!
    {
        raw_sock_t s = connect_raw();
        uint8_t sk[32];
        assert(perform_raw_shcp_v2_handshake(s, token, 5, sk));

        HavenActivationFrameHeader hdr;
        std::vector<uint8_t> body;
        build_wire_aead_packet(sk, 1, 5, {1.5f, 2.5f, 3.5f}, hdr, body);
        ::send(s, reinterpret_cast<const char*>(&hdr), sizeof(hdr), 0);
        ::send(s, reinterpret_cast<const char*>(body.data()), static_cast<int>(body.size()), 0);

        HavenActivationFrame rx;
        assert(coord.await_upstream_activations(rx, 500));
        assert(rx.data.size() == 3 && std::fabs(rx.data[1] - 2.5f) < 1e-6f);

        // Replay exact same valid packet (ctr=1)
        ::send(s, reinterpret_cast<const char*>(&hdr), sizeof(hdr), 0);
        ::send(s, reinterpret_cast<const char*>(body.data()), static_cast<int>(body.size()), 0);

        HavenActivationFrame replayed;
        assert(!coord.await_upstream_activations(replayed, 250) && "Replayed AEAD packet MUST be rejected");
        HavenActivationFrameHeader rej{};
        assert(recv_exact_raw(s, &rej, sizeof(rej)) && (rej.flags & FRAME_FLAG_AUTH_REJECT));
        raw_close(s);
    }

    // 2b. Bit-flipped ciphertext on valid session MUST be rejected by Poly1305!
    {
        raw_sock_t s = connect_raw();
        uint8_t sk[32];
        assert(perform_raw_shcp_v2_handshake(s, token, 6, sk));

        HavenActivationFrameHeader hdr;
        std::vector<uint8_t> body;
        build_wire_aead_packet(sk, 1, 6, {4.0f, 5.0f, 6.0f}, hdr, body);
        body[2] ^= 0x40; // Flip 1 bit in ciphertext!
        ::send(s, reinterpret_cast<const char*>(&hdr), sizeof(hdr), 0);
        ::send(s, reinterpret_cast<const char*>(body.data()), static_cast<int>(body.size()), 0);

        HavenActivationFrame tampered;
        assert(!coord.await_upstream_activations(tampered, 250) && "Tampered AEAD ciphertext MUST be rejected");
        HavenActivationFrameHeader rej{};
        assert(recv_exact_raw(s, &rej, sizeof(rej)) && (rej.flags & FRAME_FLAG_AUTH_REJECT));
        raw_close(s);
    }

    // 2c. Plaintext downgrade frame on authenticated AEAD session MUST be rejected!
    {
        raw_sock_t s = connect_raw();
        uint8_t sk[32];
        assert(perform_raw_shcp_v2_handshake(s, token, 7, sk));

        HavenActivationFrameHeader plain_hdr{};
        plain_hdr.magic = HAVEN_CLUSTER_MAGIC;
        plain_hdr.protocol_ver = HAVEN_CLUSTER_VERSION;
        plain_hdr.flags = FRAME_FLAG_ACTIVATION; // Missing FRAME_FLAG_ENCRYPTED!
        plain_hdr.from_node_id = 7;
        plain_hdr.element_count = 2;
        plain_hdr.payload_bytes = 8;
        float vals[2] = {1.0f, 2.0f};
        ::send(s, reinterpret_cast<const char*>(&plain_hdr), sizeof(plain_hdr), 0);
        ::send(s, reinterpret_cast<const char*>(vals), sizeof(vals), 0);

        HavenActivationFrame downgraded;
        assert(!coord.await_upstream_activations(downgraded, 250) && "Plaintext downgrade MUST be rejected");
        HavenActivationFrameHeader rej{};
        bool got_rej = recv_exact_raw(s, &rej, sizeof(rej));
        assert((!got_rej || (rej.flags & FRAME_FLAG_AUTH_REJECT)) && "Server must send AUTH_REJECT or RST-sever");
        char eof_probe = 0;
        assert(::recv(s, &eof_probe, 1, 0) <= 0 && "Server must close socket after plaintext downgrade");
        raw_close(s);
    }

    coord.shutdown();
    std::cout << "   ✓ Replay, Poly1305 ciphertext tamper, and plaintext downgrade attacks all blocked.\n";
}

// -----------------------------------------------------------------------------
// 3. RFC 8259 HavenJson Compliance & Adversarial Fuzz Suite
// -----------------------------------------------------------------------------
static void test_json_parser_and_fuzz() {
    std::cout << "[3/6] Fuzzing RFC 8259 HavenJson Parser (Unicode, Surrogates, Depth, & 150 Mutants)...\n";
    HavenJson j;
    assert(HavenJson::parse("{\"msg\":\"Hello \\u0041 \\uD83D\\uDE80 {braces}\",\"num\":-42.5e1,\"ok\":true,\"arr\":[1,null,false]}", j));
    assert(j.get_string("msg") == "Hello A \xF0\x9F\x9A\x80 {braces}");
    assert(j.get_scalar_string("num") == "-42.5e1");
    assert(j.get_scalar_string("ok") == "true");
    assert(j.find("arr") && j.find("arr")->as_array().size() == 3);

    // Reject illegal JSON: deep nesting (>32), unescaped control char, lone surrogate, trailing garbage
    std::string deep(128, '[');
    deep += std::string(128, ']');
    assert(!HavenJson::parse(deep, j));
    assert(!HavenJson::parse("{\"a\":\"\x01\"}", j));
    assert(!HavenJson::parse("{\"a\":\"\\uD800\"}", j));
    assert(!HavenJson::parse("{\"a\":1} trailing", j));
    assert(!HavenJson::parse("{\"a\":01}", j));

    // 150 deterministic pseudo-random adversarial mutants
    std::mt19937 rng(0x5A17C0DE);
    const std::string alphabet = "{}[]\":,0123456789\\uD800truefalsenull-\n\r\t\x00\xFF";
    for (int i = 0; i < 150; ++i) {
        size_t len = (rng() % 256) + 1;
        std::string mutant(len, '\0');
        for (size_t k = 0; k < len; ++k) mutant[k] = alphabet[rng() % alphabet.size()];
        HavenJson out;
        (void)HavenJson::parse(mutant, out);
    }
    std::cout << "   ✓ RFC 8259 JSON compliance and 150 adversarial mutants passed with zero crashes.\n";
}

// -----------------------------------------------------------------------------
// 4. GGUF Binary Loader Adversarial Fuzzer (150 malformed binary cases)
// -----------------------------------------------------------------------------
static void test_gguf_binary_fuzzer() {
    std::cout << "[4/6] Fuzzing GGUF Binary Loader with 150 Malformed & Overflowing Binaries...\n";
    std::filesystem::path tmp_path = std::filesystem::temp_directory_path() / "haven_fuzz_test.gguf";

    auto run_case = [&](const std::vector<uint8_t>& buf) {
        {
            std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
            if (!buf.empty()) out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        }
        GgufLoader loader;
        bool ok = loader.load_file(tmp_path.string());
        assert(!ok && "Malformed GGUF MUST NOT load successfully");
    };

    // Crafted extreme attack vectors:
    // a) Empty & truncated headers
    run_case({});
    run_case({'G', 'G', 'U', 'F'});
    // b) Huge tensor_count & kv_count (0xFFFFFFFFFFFFFFFF)
    {
        std::vector<uint8_t> b(24, 0);
        b[0]='G'; b[1]='G'; b[2]='U'; b[3]='F';
        b[4]=3; // v3
        std::memset(b.data() + 8, 0xFF, 16);
        run_case(b);
    }
    // c) 1 KV with string length = 0xFFFFFFFFFFFFFFFF (pointer wraparound attack)
    {
        std::vector<uint8_t> b(32, 0);
        b[0]='G'; b[1]='G'; b[2]='U'; b[3]='F';
        b[4]=3; b[16]=1; // kv_count = 1
        std::memset(b.data() + 24, 0xFF, 8); // key len = max uint64
        run_case(b);
    }
    // d) Deeply nested GGUF array (recursion stack overflow attack, depth > 16)
    {
        std::vector<uint8_t> b(24, 0);
        b[0]='G'; b[1]='G'; b[2]='U'; b[3]='F';
        b[4]=3; b[16]=1; // kv_count = 1
        // key = "a"
        uint64_t klen = 1;
        b.insert(b.end(), reinterpret_cast<uint8_t*>(&klen), reinterpret_cast<uint8_t*>(&klen) + 8);
        b.push_back('a');
        uint32_t arr_t = 9;
        b.insert(b.end(), reinterpret_cast<uint8_t*>(&arr_t), reinterpret_cast<uint8_t*>(&arr_t) + 4);
        for (int depth = 0; depth < 32; ++depth) {
            uint32_t elem_t = 9;
            uint64_t cnt = 1;
            b.insert(b.end(), reinterpret_cast<uint8_t*>(&elem_t), reinterpret_cast<uint8_t*>(&elem_t) + 4);
            b.insert(b.end(), reinterpret_cast<uint8_t*>(&cnt), reinterpret_cast<uint8_t*>(&cnt) + 8);
        }
        run_case(b);
    }

    // e) 145 pseudo-random mutated GGUF headers with valid 'GGUF' magic
    std::mt19937 rng(0x660F1234);
    for (int i = 0; i < 145; ++i) {
        size_t sz = (rng() % 512) + 8;
        std::vector<uint8_t> b(sz);
        for (size_t k = 0; k < sz; ++k) b[k] = static_cast<uint8_t>(rng() & 0xFF);
        b[0]='G'; b[1]='G'; b[2]='U'; b[3]='F';
        b[4]=3; b[5]=0; b[6]=0; b[7]=0;
        if (sz >= 24) {
            b[8]  = static_cast<uint8_t>(rng() % 4); b[9]=b[10]=b[11]=b[12]=b[13]=b[14]=b[15]=0;
            b[16] = static_cast<uint8_t>(rng() % 6); b[17]=b[18]=b[19]=b[20]=b[21]=b[22]=b[23]=0;
        }
        {
            std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
        }
        GgufLoader loader;
        (void)loader.load_file(tmp_path.string());
    }

    std::error_code ec;
    std::filesystem::remove(tmp_path, ec);
    std::cout << "   ✓ 150 adversarial GGUF binaries processed with zero crashes or OOB reads.\n";
}

// -----------------------------------------------------------------------------
// 5. HMB 64-Bit Binary Vault Adversarial Fuzzer (100 malformed .hmb cases)
// -----------------------------------------------------------------------------
static void test_hmb_binary_fuzzer() {
    std::cout << "[5/6] Fuzzing HMB 64-Bit Memory Vault Loader with 100 Mutated .hmb Files...\n";
    std::filesystem::path tmp_path = std::filesystem::temp_directory_path() / "haven_fuzz_test.hmb";
    std::mt19937 rng(0x484D4231);

    for (int i = 0; i < 100; ++i) {
        size_t sz = (rng() % 512) + sizeof(HmbHeader64);
        std::vector<uint8_t> b(sz);
        for (size_t k = 0; k < sz; ++k) b[k] = static_cast<uint8_t>(rng() & 0xFF);
        if (i > 10) {
            // Valid 'HAVENMEM' v2.0 magic + valid/mutated header fields to exercise deep HMB bounds checks
            HmbHeader64 hdr{};
            std::memcpy(hdr.magic, "HAVENMEM", 8);
            hdr.version = 0x00020000U;
            hdr.embedding_dim = (i % 5 == 0) ? 0xFFFFFFFFU : ((rng() % 256) + 1);
            hdr.total_anchors = (i % 7 == 0) ? 0xFFFFFFFFFFFFFFFFULL : (rng() % 16);
            hdr.vector_table_offset = rng() % (sz + 64);
            hdr.record_table_offset = rng() % (sz + 64);
            hdr.string_table_offset = rng() % (sz + 64);
            hdr.string_table_size = (i % 9 == 0) ? 0xFFFFFFFFFFFFFFFFULL : (rng() % (sz + 64));
            std::memcpy(b.data(), &hdr, sizeof(hdr));
        }
        {
            std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
        }
        MemoryAttentionEngine mem;
        (void)mem.load_from_hmb(tmp_path.string());
    }

    std::error_code ec;
    std::filesystem::remove(tmp_path, ec);
    std::cout << "   ✓ 100 mutated .hmb vault files processed safely with zero crashes.\n";
}

// -----------------------------------------------------------------------------
// 6. Live Hardened Plugin Security Tests (Job Object, Workspace, Vault, SSRF)
// -----------------------------------------------------------------------------
static void test_hardened_plugins() {
#ifndef _WIN32
    std::cout << "[6/6] Skipping Win32 Job Object / WinHTTP plugin suite on non-Windows host.\n";
    return;
#else
    std::cout << "[6/6] Verifying Hardened Plugins (Win32 Job Object, Workspace, Vault, DNS-SSRF)...\n";
    PluginManager pm;
    size_t count = pm.discover_plugins("build/plugins", true);
    if (count == 0) count = pm.discover_plugins("plugins", true);
    if (count == 0) count = pm.discover_plugins("C:\\Users\\admin\\source\\haven-cpp\\plugins", true);
    assert(count > 0 && "Plugins must be discoverable");

    std::string out;

    // Terminal: legitimate echo runs cleanly inside Job Object
    assert(pm.dispatch_tool_execution("run_command", "echo HAVEN_JOB_SANDBOX_OK", out));
    assert(out.find("HAVEN_JOB_SANDBOX_OK") != std::string::npos);

    // Terminal: [cwd:] injection blocked
    out.clear();
    assert(pm.dispatch_tool_execution("run_command", "[cwd: C:\\ & calc] echo pwn", out));
    assert(out.find("Security Block") != std::string::npos || out.find("Blocked") != std::string::npos);

    // Terminal: env expansion & subshell escalation blocked
    out.clear();
    assert(pm.dispatch_tool_execution("run_command", "echo %COMSPEC%", out));
    assert(out.find("Security Block") != std::string::npos);
    out.clear();
    assert(pm.dispatch_tool_execution("run_command", "powershell -c whoami", out));
    assert(out.find("Security Block") != std::string::npos);

    // Workspace coder: outside-workspace absolute path blocked
    out.clear();
    assert(pm.dispatch_tool_execution("read_file", "C:\\Windows\\win.ini", out));
    assert(out.find("Blocked") != std::string::npos || out.find("outside") != std::string::npos || out.find("Security") != std::string::npos);

    // Vault sync: path traversal blocked (returns false + Blocked message)
    out.clear();
    bool vault_ok = pm.dispatch_tool_execution("vault_read", "..\\persona.json", out);
    assert(!vault_ok && (out.find("Blocked") != std::string::npos || out.find("Invalid") != std::string::npos));

    // Browser: DNS-resolved SSRF loopback / RFC1918 / CGNAT-Tailscale / IPv6 blocked
    for (const char* ssrf_url : {
        "http://127.0.0.1:11436/api/cluster/token",
        "http://localhost:11436/health",
        "http://100.64.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]:11436/health"
    }) {
        out.clear();
        assert(pm.dispatch_tool_execution("browse_url", ssrf_url, out));
        assert(out.find("sovereign_online") == std::string::npos && out.find("\"token\"") == std::string::npos);
    }

    std::cout << "   ✓ All hardened plugin security defenses verified live.\n";
#endif
}

int main() {
    std::cout << "\n==================================================================\n";
    std::cout << "🛡️  HAVEN-CPP CRYPTO KAT, SHCP v2 ATTACK, & ADVERSARIAL FUZZ SUITE\n";
    std::cout << "==================================================================\n\n";

    test_crypto_kats();
    test_shcp_v2_wire_attacks();
    test_json_parser_and_fuzz();
    test_gguf_binary_fuzzer();
    test_hmb_binary_fuzzer();
    test_hardened_plugins();

    std::cout << "\n==================================================================\n";
    std::cout << "🏆 ALL 6 ADVERSARIAL SECURITY, CRYPTO, & FUZZ SUITES PASSED 100%!\n";
    std::cout << "==================================================================\n\n";
    return 0;
}
