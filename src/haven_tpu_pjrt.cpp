#include "haven/haven_tpu_pjrt.h"
#include "haven/haven_avx2.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>

#ifndef _WIN32
#include <dlfcn.h>
#include "haven/third_party/pjrt_c_api.h"
#endif

namespace haven {

uint16_t HavenTpuPjrtBackend::fp32_to_bf16(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7FFFFFFFU) > 0x7F800000U) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040U); // Quiet NaN
    }
    const uint32_t lsb = (bits >> 16) & 1U;
    const uint32_t rounding_bias = 0x7FFFU + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

float HavenTpuPjrtBackend::bf16_to_fp32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void HavenTpuPjrtBackend::convert_f32_to_bf16(const float* src, uint16_t* dst, size_t n) {
    if (!src || !dst) return;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fp32_to_bf16(src[i]);
    }
}

void HavenTpuPjrtBackend::convert_bf16_to_f32(const uint16_t* src, float* dst, size_t n) {
    if (!src || !dst) return;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = bf16_to_fp32(src[i]);
    }
}

HavenTpuPjrtBackend::HavenTpuPjrtBackend() = default;

HavenTpuPjrtBackend::~HavenTpuPjrtBackend() {
    shutdown();
}

#ifdef _WIN32

bool HavenTpuPjrtBackend::initialize(const std::string&) {
    last_error_ = "Native Cloud TPU PJRT (libtpu.so) is only available on Linux Cloud TPU hosts.";
    return false;
}

void HavenTpuPjrtBackend::shutdown() {
    initialized_ = false;
}

PJRT_Buffer* HavenTpuPjrtBackend::upload_bf16_2d(const uint16_t*, int64_t, int64_t, int) { return nullptr; }
PJRT_Buffer* HavenTpuPjrtBackend::upload_f32_2d(const float*, int64_t, int64_t, int) { return nullptr; }
PJRT_Buffer* HavenTpuPjrtBackend::upload_bf16_nd(const uint16_t*, const std::vector<int64_t>&, int) { return nullptr; }
PJRT_Buffer* HavenTpuPjrtBackend::upload_f32_nd(const float*, const std::vector<int64_t>&, int) { return nullptr; }
PJRT_Buffer* HavenTpuPjrtBackend::upload_s8_nd(const int8_t*, const std::vector<int64_t>&, int) { return nullptr; }
PJRT_Buffer* HavenTpuPjrtBackend::upload_f8e4m3fn_nd(const uint8_t*, const std::vector<int64_t>&, int) { return nullptr; }
PJRT_Buffer* HavenTpuPjrtBackend::upload_i32_nd(const int32_t*, const std::vector<int64_t>&, int) { return nullptr; }
bool HavenTpuPjrtBackend::download_f32(PJRT_Buffer*, float*, size_t) { return false; }
bool HavenTpuPjrtBackend::download_bf16(PJRT_Buffer*, uint16_t*, size_t) { return false; }
bool HavenTpuPjrtBackend::download_i32(PJRT_Buffer*, int32_t*, size_t) { return false; }
void HavenTpuPjrtBackend::destroy_buffer(PJRT_Buffer*) {}
PJRT_LoadedExecutable* HavenTpuPjrtBackend::compile_stablehlo(const std::string&, int) { return nullptr; }
PJRT_LoadedExecutable* HavenTpuPjrtBackend::compile_stablehlo_replicated(const std::string&, int) { return nullptr; }
std::vector<int> HavenTpuPjrtBackend::get_executable_device_indices(PJRT_LoadedExecutable*) { return {}; }
void HavenTpuPjrtBackend::destroy_executable(PJRT_LoadedExecutable*) {}
bool HavenTpuPjrtBackend::execute_1d(PJRT_LoadedExecutable*, int, const std::vector<PJRT_Buffer*>&, size_t, std::vector<PJRT_Buffer*>&) { return false; }
bool HavenTpuPjrtBackend::execute_replicated(PJRT_LoadedExecutable*, int, const std::vector<std::vector<PJRT_Buffer*>>&, size_t, std::vector<std::vector<PJRT_Buffer*>>&) { return false; }
bool HavenTpuPjrtBackend::upload_gguf_weight_matrix(const TensorDesc&, int, int, TpuMatrixHandle&, int) { return false; }
void HavenTpuPjrtBackend::free_matrix(TpuMatrixHandle& handle) { handle = {}; }
bool HavenTpuPjrtBackend::matmul_resident_nt(const float*, int, const TpuMatrixHandle&, float*) { return false; }
bool HavenTpuPjrtBackend::fused_geglu_ffn(const float*, int, const TpuMatrixHandle&, const TpuMatrixHandle&, const TpuMatrixHandle&, float*) { return false; }
TpuGemmBenchResult HavenTpuPjrtBackend::benchmark_bf16_gemm(int M, int N, int K, int iterations, int num_devices) {
    TpuGemmBenchResult res;
    res.M = M; res.N = N; res.K = K; res.iterations = iterations; res.num_devices = num_devices;
    res.error = "Native Cloud TPU PJRT is unavailable on Windows.";
    return res;
}
void HavenTpuPjrtBackend::refresh_memory_stats() {}
std::string HavenTpuPjrtBackend::consume_error(void*) { return ""; }
bool HavenTpuPjrtBackend::await_and_destroy_event(PJRT_Event*) { return false; }
PJRT_LoadedExecutable* HavenTpuPjrtBackend::get_or_compile_matmul_nt(int, int, int, int) { return nullptr; }
PJRT_LoadedExecutable* HavenTpuPjrtBackend::get_or_compile_geglu_ffn(int, int, int, int) { return nullptr; }

#else // Linux / POSIX Cloud TPU Host

namespace {

using GetPjrtApiFn = const PJRT_Api* (*)();

// Minimal protobuf varint encoder for xla::CompileOptionsProto:
//   CompileOptionsProto { ExecutableBuildOptionsProto executable_build_options = 3; }
//   ExecutableBuildOptionsProto { int64 device_ordinal = 2; int64 num_replicas = 4; int64 num_partitions = 5; }
static void append_varint(std::string& out, uint64_t v) {
    while (v >= 0x80ULL) {
        out.push_back(static_cast<char>((v & 0x7FULL) | 0x80ULL));
        v >>= 7;
    }
    out.push_back(static_cast<char>(v));
}

static void append_int64_field(std::string& out, uint32_t field_num, int64_t val) {
    append_varint(out, (static_cast<uint64_t>(field_num) << 3) | 0ULL);
    append_varint(out, static_cast<uint64_t>(val));
}

static std::string encode_compile_options_proto(int device_ordinal, int num_replicas = 1, int num_partitions = 1, bool portable = false) {
    std::string build_opts;
    if (!portable && device_ordinal >= 0) {
        append_int64_field(build_opts, 1, device_ordinal); // ExecutableBuildOptionsProto.device_ordinal = 1
    }
    if (num_replicas > 0) {
        append_int64_field(build_opts, 4, num_replicas);   // num_replicas = 4
    }
    if (num_partitions > 0) {
        append_int64_field(build_opts, 5, num_partitions); // num_partitions = 5
    }
    std::string compile_opts;
    if (!build_opts.empty()) {
        append_varint(compile_opts, (3ULL << 3) | 2ULL); // CompileOptionsProto.executable_build_options = 3
        append_varint(compile_opts, build_opts.size());
        compile_opts.append(build_opts);
    }
    if (portable) {
        append_int64_field(compile_opts, 4, 1);          // CompileOptionsProto.compile_portable_executable = 4
    }
    return compile_opts;
}

static std::vector<std::string> discover_libtpu_candidates(const std::string& override_path) {
    std::vector<std::string> paths;
    if (!override_path.empty()) paths.push_back(override_path);
    if (const char* env_p = std::getenv("HAVEN_LIBTPU_PATH"); env_p && env_p[0] != '\0') {
        paths.emplace_back(env_p);
    }
    if (const char* env_p = std::getenv("TPU_LIBRARY_PATH"); env_p && env_p[0] != '\0') {
        paths.emplace_back(env_p);
    }
    paths.emplace_back("libtpu.so");
    paths.emplace_back("/lib/libtpu.so");
    paths.emplace_back("/usr/lib/libtpu.so");
    paths.emplace_back("/usr/local/lib/libtpu.so");

    // Probe standard Python site-packages / dist-packages locations on GCP TPU VMs
    const std::vector<std::string> py_roots = {
        "/usr/local/lib",
        "/usr/lib",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "/root") + "/.local/lib"
    };
    std::error_code ec;
    for (const auto& root : py_roots) {
        if (!std::filesystem::exists(root, ec)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
            if (!entry.is_directory(ec)) continue;
            const std::string name = entry.path().filename().string();
            if (name.rfind("python3", 0) != 0) continue;
            for (const char* sub : {"dist-packages/libtpu/libtpu.so", "site-packages/libtpu/libtpu.so"}) {
                auto cand = entry.path() / sub;
                if (std::filesystem::exists(cand, ec)) {
                    paths.push_back(cand.string());
                }
            }
        }
    }
    return paths;
}

} // namespace

std::string HavenTpuPjrtBackend::consume_error(void* pjrt_error_ptr) {
    auto* err = static_cast<PJRT_Error*>(pjrt_error_ptr);
    if (!err || !api_) return "";
    PJRT_Error_Message_Args margs{};
    margs.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    margs.error = err;
    api_->PJRT_Error_Message(&margs);
    std::string msg = (margs.message && margs.message_size > 0)
        ? std::string(margs.message, margs.message_size)
        : "Unknown PJRT error";

    PJRT_Error_Destroy_Args dargs{};
    dargs.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    dargs.error = err;
    api_->PJRT_Error_Destroy(&dargs);
    return msg;
}

bool HavenTpuPjrtBackend::await_and_destroy_event(PJRT_Event* ev) {
    if (!ev || !api_) return true;
    PJRT_Event_Await_Args aargs{};
    aargs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    aargs.event = ev;
    PJRT_Error* err = api_->PJRT_Event_Await(&aargs);
    bool ok = true;
    if (err) {
        last_error_ = consume_error(err);
        ok = false;
    }

    PJRT_Event_Destroy_Args dargs{};
    dargs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    dargs.event = ev;
    if (PJRT_Error* derr = api_->PJRT_Event_Destroy(&dargs)) {
        (void)consume_error(derr);
    }
    return ok;
}

bool HavenTpuPjrtBackend::initialize(const std::string& libtpu_path_override) {
    if (initialized_) return true;
    last_error_.clear();

    auto candidates = discover_libtpu_candidates(libtpu_path_override);
    std::string loaded_path;
    for (const auto& p : candidates) {
        void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (h) {
            dl_handle_ = h;
            loaded_path = p;
            break;
        }
    }
    if (!dl_handle_) {
        last_error_ = "Failed to dlopen libtpu.so from default search paths.";
        return false;
    }

    auto get_api = reinterpret_cast<GetPjrtApiFn>(dlsym(dl_handle_, "GetPjrtApi"));
    if (!get_api) {
        last_error_ = "Loaded " + loaded_path + " but symbol GetPjrtApi not found.";
        shutdown();
        return false;
    }

    api_ = get_api();
    if (!api_) {
        last_error_ = "GetPjrtApi() returned nullptr.";
        shutdown();
        return false;
    }

    pjrt_major_ = api_->pjrt_api_version.major_version;
    pjrt_minor_ = api_->pjrt_api_version.minor_version;

    // 1. One-time PJRT plugin initialization
    if (api_->PJRT_Plugin_Initialize) {
        PJRT_Plugin_Initialize_Args init_args{};
        init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
        if (PJRT_Error* err = api_->PJRT_Plugin_Initialize(&init_args)) {
            last_error_ = "PJRT_Plugin_Initialize failed: " + consume_error(err);
            shutdown();
            return false;
        }
    }

    // 2. Create PJRT Client
    PJRT_Client_Create_Args create_args{};
    create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    if (PJRT_Error* err = api_->PJRT_Client_Create(&create_args)) {
        last_error_ = "PJRT_Client_Create failed: " + consume_error(err);
        shutdown();
        return false;
    }
    client_ = create_args.client;
    if (!client_) {
        last_error_ = "PJRT_Client_Create returned null client.";
        shutdown();
        return false;
    }

    // 3. Query platform metadata
    {
        PJRT_Client_PlatformName_Args pargs{};
        pargs.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
        pargs.client = client_;
        if (PJRT_Error* err = api_->PJRT_Client_PlatformName(&pargs)) {
            (void)consume_error(err);
        } else if (pargs.platform_name && pargs.platform_name_size > 0) {
            platform_name_.assign(pargs.platform_name, pargs.platform_name_size);
        }

        PJRT_Client_PlatformVersion_Args vargs{};
        vargs.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
        vargs.client = client_;
        if (PJRT_Error* err = api_->PJRT_Client_PlatformVersion(&vargs)) {
            (void)consume_error(err);
        } else if (vargs.platform_version && vargs.platform_version_size > 0) {
            platform_version_.assign(vargs.platform_version, vargs.platform_version_size);
        }
    }

    // 4. Enumerate all addressable TPU devices (e.g., all 8 Trillium TensorCores on v6e-8)
    {
        PJRT_Client_AddressableDevices_Args dargs{};
        dargs.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
        dargs.client = client_;
        if (PJRT_Error* err = api_->PJRT_Client_AddressableDevices(&dargs)) {
            last_error_ = "PJRT_Client_AddressableDevices failed: " + consume_error(err);
            shutdown();
            return false;
        }
        devices_.assign(dargs.addressable_devices, dargs.addressable_devices + dargs.num_addressable_devices);
        device_infos_.clear();
        device_infos_.reserve(devices_.size());

        for (PJRT_Device* dev : devices_) {
            TpuDeviceInfo info;
            PJRT_Device_GetDescription_Args gd{};
            gd.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
            gd.device = dev;
            if (PJRT_Error* err = api_->PJRT_Device_GetDescription(&gd)) {
                (void)consume_error(err);
            } else if (gd.device_description) {
                PJRT_DeviceDescription_Id_Args id_args{};
                id_args.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
                id_args.device_description = gd.device_description;
                if (PJRT_Error* e = api_->PJRT_DeviceDescription_Id(&id_args)) (void)consume_error(e);
                else info.id = id_args.id;

                PJRT_DeviceDescription_Kind_Args k_args{};
                k_args.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
                k_args.device_description = gd.device_description;
                if (PJRT_Error* e = api_->PJRT_DeviceDescription_Kind(&k_args)) (void)consume_error(e);
                else if (k_args.device_kind && k_args.device_kind_size > 0)
                    info.kind.assign(k_args.device_kind, k_args.device_kind_size);

                PJRT_DeviceDescription_DebugString_Args dbg_args{};
                dbg_args.struct_size = PJRT_DeviceDescription_DebugString_Args_STRUCT_SIZE;
                dbg_args.device_description = gd.device_description;
                if (PJRT_Error* e = api_->PJRT_DeviceDescription_DebugString(&dbg_args)) (void)consume_error(e);
                else if (dbg_args.debug_string && dbg_args.debug_string_size > 0)
                    info.debug_string.assign(dbg_args.debug_string, dbg_args.debug_string_size);
            }

            PJRT_Device_LocalHardwareId_Args hw_args{};
            hw_args.struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE;
            hw_args.device = dev;
            if (PJRT_Error* e = api_->PJRT_Device_LocalHardwareId(&hw_args)) (void)consume_error(e);
            else info.local_hardware_id = hw_args.local_hardware_id;

            device_infos_.push_back(std::move(info));
        }
    }

    initialized_ = true;
    refresh_memory_stats();
    return true;
}

void HavenTpuPjrtBackend::refresh_memory_stats() {
    if (!initialized_ || !api_ || !api_->PJRT_Device_MemoryStats) return;
    for (size_t i = 0; i < devices_.size() && i < device_infos_.size(); ++i) {
        PJRT_Device_MemoryStats_Args ms{};
        ms.struct_size = PJRT_STRUCT_SIZE(PJRT_Device_MemoryStats_Args, bytes_limit_is_set);
        ms.device = devices_[i];
        if (PJRT_Error* e = api_->PJRT_Device_MemoryStats(&ms)) {
            (void)consume_error(e);
        } else {
            device_infos_[i].hbm_bytes_in_use = ms.bytes_in_use;
            if (ms.bytes_limit_is_set) {
                device_infos_[i].hbm_bytes_limit = ms.bytes_limit;
            }
        }
    }
}

void HavenTpuPjrtBackend::shutdown() {
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (auto& [k, exec] : exec_cache_) {
            destroy_executable(exec);
        }
        exec_cache_.clear();
    }

    if (client_ && api_ && api_->PJRT_Client_Destroy) {
        PJRT_Client_Destroy_Args dargs{};
        dargs.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        dargs.client = client_;
        if (PJRT_Error* err = api_->PJRT_Client_Destroy(&dargs)) {
            (void)consume_error(err);
        }
    }
    client_ = nullptr;
    devices_.clear();
    device_infos_.clear();
    api_ = nullptr;
    if (dl_handle_) {
        dlclose(dl_handle_);
        dl_handle_ = nullptr;
    }
    initialized_ = false;
}

PJRT_Buffer* HavenTpuPjrtBackend::upload_bf16_nd(const uint16_t* host_data, const std::vector<int64_t>& dims, int device_index) {
    if (!initialized_ || !host_data || device_index < 0 || static_cast<size_t>(device_index) >= devices_.size()) {
        return nullptr;
    }
    PJRT_Client_BufferFromHostBuffer_Args args{};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = client_;
    args.data = host_data;
    args.type = PJRT_Buffer_Type_BF16;
    args.dims = dims.empty() ? nullptr : dims.data();
    args.num_dims = dims.size();
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    args.device = devices_[static_cast<size_t>(device_index)];

    if (PJRT_Error* err = api_->PJRT_Client_BufferFromHostBuffer(&args)) {
        last_error_ = "BufferFromHostBuffer(BF16) failed: " + consume_error(err);
        return nullptr;
    }
    if (args.done_with_host_buffer) {
        await_and_destroy_event(args.done_with_host_buffer);
    }
    return args.buffer;
}

PJRT_Buffer* HavenTpuPjrtBackend::upload_f32_nd(const float* host_data, const std::vector<int64_t>& dims, int device_index) {
    if (!initialized_ || !host_data || device_index < 0 || static_cast<size_t>(device_index) >= devices_.size()) {
        return nullptr;
    }
    PJRT_Client_BufferFromHostBuffer_Args args{};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = client_;
    args.data = host_data;
    args.type = PJRT_Buffer_Type_F32;
    args.dims = dims.empty() ? nullptr : dims.data();
    args.num_dims = dims.size();
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    args.device = devices_[static_cast<size_t>(device_index)];

    if (PJRT_Error* err = api_->PJRT_Client_BufferFromHostBuffer(&args)) {
        last_error_ = "BufferFromHostBuffer(F32) failed: " + consume_error(err);
        return nullptr;
    }
    if (args.done_with_host_buffer) {
        await_and_destroy_event(args.done_with_host_buffer);
    }
    return args.buffer;
}

PJRT_Buffer* HavenTpuPjrtBackend::upload_s8_nd(const int8_t* host_data, const std::vector<int64_t>& dims, int device_index) {
    if (!initialized_ || !host_data || device_index < 0 || static_cast<size_t>(device_index) >= devices_.size()) {
        return nullptr;
    }
    PJRT_Client_BufferFromHostBuffer_Args args{};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = client_;
    args.data = host_data;
    args.type = PJRT_Buffer_Type_S8;
    args.dims = dims.empty() ? nullptr : dims.data();
    args.num_dims = dims.size();
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    args.device = devices_[static_cast<size_t>(device_index)];

    if (PJRT_Error* err = api_->PJRT_Client_BufferFromHostBuffer(&args)) {
        last_error_ = "BufferFromHostBuffer(S8) failed: " + consume_error(err);
        return nullptr;
    }
    if (args.done_with_host_buffer) {
        await_and_destroy_event(args.done_with_host_buffer);
    }
    return args.buffer;
}

PJRT_Buffer* HavenTpuPjrtBackend::upload_f8e4m3fn_nd(const uint8_t* host_data, const std::vector<int64_t>& dims, int device_index) {
    if (!initialized_ || !host_data || device_index < 0 || static_cast<size_t>(device_index) >= devices_.size()) {
        return nullptr;
    }
    PJRT_Client_BufferFromHostBuffer_Args args{};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = client_;
    args.data = host_data;
    args.type = PJRT_Buffer_Type_F8E4M3FN;
    args.dims = dims.empty() ? nullptr : dims.data();
    args.num_dims = dims.size();
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    args.device = devices_[static_cast<size_t>(device_index)];

    if (PJRT_Error* err = api_->PJRT_Client_BufferFromHostBuffer(&args)) {
        last_error_ = "BufferFromHostBuffer(F8E4M3FN) failed: " + consume_error(err);
        return nullptr;
    }
    if (args.done_with_host_buffer) {
        await_and_destroy_event(args.done_with_host_buffer);
    }
    return args.buffer;
}

PJRT_Buffer* HavenTpuPjrtBackend::upload_i32_nd(const int32_t* host_data, const std::vector<int64_t>& dims, int device_index) {
    if (!initialized_ || !host_data || device_index < 0 || static_cast<size_t>(device_index) >= devices_.size()) {
        return nullptr;
    }
    PJRT_Client_BufferFromHostBuffer_Args args{};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = client_;
    args.data = host_data;
    args.type = PJRT_Buffer_Type_S32;
    args.dims = dims.empty() ? nullptr : dims.data();
    args.num_dims = dims.size();
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    args.device = devices_[static_cast<size_t>(device_index)];

    if (PJRT_Error* err = api_->PJRT_Client_BufferFromHostBuffer(&args)) {
        last_error_ = "BufferFromHostBuffer(S32) failed: " + consume_error(err);
        return nullptr;
    }
    if (args.done_with_host_buffer) {
        await_and_destroy_event(args.done_with_host_buffer);
    }
    return args.buffer;
}

PJRT_Buffer* HavenTpuPjrtBackend::upload_bf16_2d(const uint16_t* host_data, int64_t rows, int64_t cols, int device_index) {
    if (rows <= 0 || cols <= 0) return nullptr;
    return upload_bf16_nd(host_data, {rows, cols}, device_index);
}

PJRT_Buffer* HavenTpuPjrtBackend::upload_f32_2d(const float* host_data, int64_t rows, int64_t cols, int device_index) {
    if (rows <= 0 || cols <= 0) return nullptr;
    return upload_f32_nd(host_data, {rows, cols}, device_index);
}

bool HavenTpuPjrtBackend::download_f32(PJRT_Buffer* buf, float* host_out, size_t num_elems) {
    if (!initialized_ || !buf || !host_out || num_elems == 0) return false;
    PJRT_Buffer_ToHostBuffer_Args args{};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = buf;
    args.dst = host_out;
    args.dst_size = num_elems * sizeof(float);
    if (PJRT_Error* err = api_->PJRT_Buffer_ToHostBuffer(&args)) {
        last_error_ = "ToHostBuffer(F32) failed: " + consume_error(err);
        return false;
    }
    return await_and_destroy_event(args.event);
}

bool HavenTpuPjrtBackend::download_bf16(PJRT_Buffer* buf, uint16_t* host_out, size_t num_elems) {
    if (!initialized_ || !buf || !host_out || num_elems == 0) return false;
    PJRT_Buffer_ToHostBuffer_Args args{};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = buf;
    args.dst = host_out;
    args.dst_size = num_elems * sizeof(uint16_t);
    if (PJRT_Error* err = api_->PJRT_Buffer_ToHostBuffer(&args)) {
        last_error_ = "ToHostBuffer(BF16) failed: " + consume_error(err);
        return false;
    }
    return await_and_destroy_event(args.event);
}

bool HavenTpuPjrtBackend::download_i32(PJRT_Buffer* buf, int32_t* host_out, size_t num_elems) {
    if (!initialized_ || !buf || !host_out || num_elems == 0) return false;
    PJRT_Buffer_ToHostBuffer_Args args{};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = buf;
    args.dst = host_out;
    args.dst_size = num_elems * sizeof(int32_t);
    if (PJRT_Error* err = api_->PJRT_Buffer_ToHostBuffer(&args)) {
        last_error_ = "ToHostBuffer(S32) failed: " + consume_error(err);
        return false;
    }
    return await_and_destroy_event(args.event);
}

void HavenTpuPjrtBackend::destroy_buffer(PJRT_Buffer* buf) {
    if (!buf || !api_ || !api_->PJRT_Buffer_Destroy) return;
    PJRT_Buffer_Destroy_Args dargs{};
    dargs.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    dargs.buffer = buf;
    if (PJRT_Error* err = api_->PJRT_Buffer_Destroy(&dargs)) {
        (void)consume_error(err);
    }
}

PJRT_LoadedExecutable* HavenTpuPjrtBackend::compile_stablehlo(const std::string& mlir_code, int device_ordinal) {
    if (!initialized_ || mlir_code.empty()) return nullptr;

    std::string code_copy = mlir_code;
    static constexpr const char kFormat[] = "mlir";
    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code = code_copy.data();
    prog.code_size = code_copy.size();
    prog.format = kFormat;
    prog.format_size = 4;

    // When device_ordinal == -1, compile a portable executable usable on ANY addressable TPU chip!
    std::string compile_opts = (device_ordinal < 0)
        ? encode_compile_options_proto(-1, 1, 1, /*portable=*/true)
        : encode_compile_options_proto(device_ordinal, 1, 1, /*portable=*/false);

    PJRT_Client_Compile_Args cargs{};
    cargs.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    cargs.client = client_;
    cargs.program = &prog;
    cargs.compile_options = compile_opts.empty() ? nullptr : compile_opts.data();
    cargs.compile_options_size = compile_opts.size();

    if (PJRT_Error* err = api_->PJRT_Client_Compile(&cargs)) {
        last_error_ = "PJRT_Client_Compile failed: " + consume_error(err);
        return nullptr;
    }
    return cargs.executable;
}

PJRT_LoadedExecutable* HavenTpuPjrtBackend::compile_stablehlo_replicated(const std::string& mlir_code, int num_replicas) {
    if (!initialized_ || mlir_code.empty() || num_replicas <= 0 ||
        static_cast<size_t>(num_replicas) > devices_.size()) {
        last_error_ = "Invalid num_replicas for compile_stablehlo_replicated.";
        return nullptr;
    }

    std::string code_copy = mlir_code;
    static constexpr const char kFormat[] = "mlir";
    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code = code_copy.data();
    prog.code_size = code_copy.size();
    prog.format = kFormat;
    prog.format_size = 4;

    std::string compile_opts = encode_compile_options_proto(-1, num_replicas, 1, /*portable=*/false);

    PJRT_Client_Compile_Args cargs{};
    cargs.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    cargs.client = client_;
    cargs.program = &prog;
    cargs.compile_options = compile_opts.data();
    cargs.compile_options_size = compile_opts.size();

    if (PJRT_Error* err = api_->PJRT_Client_Compile(&cargs)) {
        last_error_ = "PJRT_Client_Compile(replicated) failed: " + consume_error(err);
        return nullptr;
    }
    return cargs.executable;
}

std::vector<int> HavenTpuPjrtBackend::get_executable_device_indices(PJRT_LoadedExecutable* exec) {
    std::vector<int> out;
    if (!initialized_ || !exec || !api_ || !api_->PJRT_LoadedExecutable_AddressableDevices) return out;

    PJRT_LoadedExecutable_AddressableDevices_Args dargs{};
    dargs.struct_size = PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
    dargs.executable = exec;
    if (PJRT_Error* err = api_->PJRT_LoadedExecutable_AddressableDevices(&dargs)) {
        (void)consume_error(err);
        return out;
    }
    out.reserve(dargs.num_addressable_devices);
    for (size_t i = 0; i < dargs.num_addressable_devices; ++i) {
        PJRT_Device* dev = dargs.addressable_devices[i];
        int found = -1;
        for (size_t j = 0; j < devices_.size(); ++j) {
            if (devices_[j] == dev) {
                found = static_cast<int>(j);
                break;
            }
        }
        out.push_back(found);
    }
    return out;
}

void HavenTpuPjrtBackend::destroy_executable(PJRT_LoadedExecutable* exec) {
    if (!exec || !api_ || !api_->PJRT_LoadedExecutable_Destroy) return;
    PJRT_LoadedExecutable_Destroy_Args dargs{};
    dargs.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    dargs.executable = exec;
    if (PJRT_Error* err = api_->PJRT_LoadedExecutable_Destroy(&dargs)) {
        (void)consume_error(err);
    }
}

bool HavenTpuPjrtBackend::execute_1d(PJRT_LoadedExecutable* exec,
                                     int device_index,
                                     const std::vector<PJRT_Buffer*>& inputs,
                                     size_t num_outputs,
                                     std::vector<PJRT_Buffer*>& outputs) {
    if (!initialized_ || !exec || device_index < 0 ||
        static_cast<size_t>(device_index) >= devices_.size() || num_outputs == 0) {
        return false;
    }

    PJRT_ExecuteOptions opts{};
    opts.struct_size = PJRT_STRUCT_SIZE(PJRT_ExecuteOptions, call_location);

    outputs.assign(num_outputs, nullptr);
    PJRT_Buffer* const* arg_list_ptr = inputs.empty() ? nullptr : inputs.data();
    PJRT_Buffer** out_list_ptr = outputs.data();
    PJRT_Event* done_ev = nullptr;

    PJRT_LoadedExecutable_Execute_Args eargs{};
    eargs.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    eargs.executable = exec;
    eargs.options = &opts;
    eargs.argument_lists = &arg_list_ptr;
    eargs.num_devices = 1;
    eargs.num_args = inputs.size();
    eargs.output_lists = &out_list_ptr;
    eargs.device_complete_events = &done_ev;
    eargs.execute_device = devices_[static_cast<size_t>(device_index)];

    if (PJRT_Error* err = api_->PJRT_LoadedExecutable_Execute(&eargs)) {
        last_error_ = "PJRT_LoadedExecutable_Execute failed: " + consume_error(err);
        return false;
    }
    if (done_ev && !await_and_destroy_event(done_ev)) {
        for (auto*& b : outputs) {
            destroy_buffer(b);
            b = nullptr;
        }
        return false;
    }
    return true;
}

bool HavenTpuPjrtBackend::execute_replicated(PJRT_LoadedExecutable* exec,
                                             int num_replicas,
                                             const std::vector<std::vector<PJRT_Buffer*>>& inputs_per_replica,
                                             size_t num_outputs,
                                             std::vector<std::vector<PJRT_Buffer*>>& outputs_per_replica) {
    if (!initialized_ || !exec || num_replicas <= 0 ||
        static_cast<size_t>(num_replicas) > devices_.size() ||
        inputs_per_replica.size() != static_cast<size_t>(num_replicas) || num_outputs == 0) {
        last_error_ = "Invalid arguments to execute_replicated.";
        return false;
    }

    const size_t R = static_cast<size_t>(num_replicas);
    const size_t num_args = inputs_per_replica[0].size();
    std::vector<PJRT_Buffer* const*> arg_ptrs(R, nullptr);
    for (size_t r = 0; r < R; ++r) {
        if (inputs_per_replica[r].size() != num_args) {
            last_error_ = "Mismatched input argument count across replicas.";
            return false;
        }
        arg_ptrs[r] = inputs_per_replica[r].empty() ? nullptr : inputs_per_replica[r].data();
    }

    outputs_per_replica.assign(R, std::vector<PJRT_Buffer*>(num_outputs, nullptr));
    std::vector<PJRT_Buffer**> out_ptrs(R, nullptr);
    for (size_t r = 0; r < R; ++r) {
        out_ptrs[r] = outputs_per_replica[r].data();
    }
    std::vector<PJRT_Event*> done_evs(R, nullptr);

    PJRT_ExecuteOptions opts{};
    opts.struct_size = PJRT_STRUCT_SIZE(PJRT_ExecuteOptions, call_location);

    PJRT_LoadedExecutable_Execute_Args eargs{};
    eargs.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    eargs.executable = exec;
    eargs.options = &opts;
    eargs.argument_lists = arg_ptrs.data();
    eargs.num_devices = R;
    eargs.num_args = num_args;
    eargs.output_lists = out_ptrs.data();
    eargs.device_complete_events = done_evs.data();
    eargs.execute_device = nullptr; // Execute across all R compiled replicas simultaneously

    if (PJRT_Error* err = api_->PJRT_LoadedExecutable_Execute(&eargs)) {
        last_error_ = "PJRT_LoadedExecutable_Execute(replicated) failed: " + consume_error(err);
        return false;
    }

    bool all_ok = true;
    for (size_t r = 0; r < R; ++r) {
        if (done_evs[r] && !await_and_destroy_event(done_evs[r])) {
            all_ok = false;
        }
    }
    if (!all_ok) {
        for (size_t r = 0; r < R; ++r) {
            for (auto*& b : outputs_per_replica[r]) {
                destroy_buffer(b);
                b = nullptr;
            }
        }
        return false;
    }
    return true;
}

bool HavenTpuPjrtBackend::upload_gguf_weight_matrix(const TensorDesc& weight, int rows, int cols,
                                                    TpuMatrixHandle& out_handle, int device_index) {
    if (!initialized_ || !weight.data || rows <= 0 || cols <= 0) return false;
    free_matrix(out_handle);

    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    std::vector<uint16_t> bf16_buf(total);

    if (weight.type == QuantType::BF16) {
        std::memcpy(bf16_buf.data(), weight.data, total * sizeof(uint16_t));
    } else {
        // Dequantize row-by-row in parallel across host cores into BF16 staging buffer
        #pragma omp parallel
        {
            std::vector<float> row_f32(static_cast<size_t>(cols));
            #pragma omp for schedule(static)
            for (int r = 0; r < rows; ++r) {
                Avx2Math::dequantize_row(row_f32.data(), weight, r, cols);
                uint16_t* dst_row = bf16_buf.data() + static_cast<size_t>(r) * static_cast<size_t>(cols);
                convert_f32_to_bf16(row_f32.data(), dst_row, static_cast<size_t>(cols));
            }
        }
    }

    PJRT_Buffer* dev_buf = upload_bf16_2d(bf16_buf.data(), rows, cols, device_index);
    if (!dev_buf) return false;

    out_handle.buffer = dev_buf;
    out_handle.rows = rows;
    out_handle.cols = cols;
    out_handle.device_index = device_index;
    out_handle.bytes = total * sizeof(uint16_t);
    return true;
}

void HavenTpuPjrtBackend::free_matrix(TpuMatrixHandle& handle) {
    if (handle.buffer) {
        destroy_buffer(handle.buffer);
        handle.buffer = nullptr;
    }
    handle.rows = 0;
    handle.cols = 0;
    handle.bytes = 0;
}

PJRT_LoadedExecutable* HavenTpuPjrtBackend::get_or_compile_matmul_nt(int batch_m, int N, int K, int device_index) {
    (void)device_index;
    std::string key = "mm_nt_" + std::to_string(batch_m) + "_" + std::to_string(N) + "_" + std::to_string(K);
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = exec_cache_.find(key);
        if (it != exec_cache_.end()) return it->second;
    }

    std::ostringstream mlir;
    mlir << "module @haven_mm_nt {\n"
         << "  func.func @main(%x: tensor<" << batch_m << "x" << K << "xbf16>, "
         << "%w: tensor<" << N << "x" << K << "xbf16>) -> tensor<" << batch_m << "x" << N << "xf32> {\n"
         << "    %0 = stablehlo.dot_general %x, %w, contracting_dims = [1] x [1] "
         << ": (tensor<" << batch_m << "x" << K << "xbf16>, tensor<" << N << "x" << K << "xbf16>) "
         << "-> tensor<" << batch_m << "x" << N << "xf32>\n"
         << "    return %0 : tensor<" << batch_m << "x" << N << "xf32>\n"
         << "  }\n"
         << "}\n";

    // Compile portable executable once; runs on any of the addressable TPU chips!
    PJRT_LoadedExecutable* exec = compile_stablehlo(mlir.str(), -1);
    if (!exec) return nullptr;

    std::lock_guard<std::mutex> lock(cache_mutex_);
    exec_cache_[key] = exec;
    return exec;
}

PJRT_LoadedExecutable* HavenTpuPjrtBackend::get_or_compile_geglu_ffn(int batch_m, int hidden_dim, int emb_dim, int device_index) {
    (void)device_index;
    std::string key = "geglu_" + std::to_string(batch_m) + "_" + std::to_string(hidden_dim) + "_" + std::to_string(emb_dim);
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = exec_cache_.find(key);
        if (it != exec_cache_.end()) return it->second;
    }

    std::ostringstream mlir;
    const std::string t_in   = "tensor<" + std::to_string(batch_m) + "x" + std::to_string(emb_dim) + "xbf16>";
    const std::string t_wg   = "tensor<" + std::to_string(hidden_dim) + "x" + std::to_string(emb_dim) + "xbf16>";
    const std::string t_wd   = "tensor<" + std::to_string(emb_dim) + "x" + std::to_string(hidden_dim) + "xbf16>";
    const std::string t_hf32 = "tensor<" + std::to_string(batch_m) + "x" + std::to_string(hidden_dim) + "xf32>";
    const std::string t_hbf  = "tensor<" + std::to_string(batch_m) + "x" + std::to_string(hidden_dim) + "xbf16>";
    const std::string t_out  = "tensor<" + std::to_string(batch_m) + "x" + std::to_string(emb_dim) + "xf32>";

    // Exact tanh-approximation GeGLU fused on TPU HBM3 + MXU:
    //   gate = X * W_gate^T, up = X * W_up^T
    //   gelu(g) = 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3)))
    //   out = bf16(gelu(gate) * up) * W_down^T
    mlir << "module @haven_fused_geglu {\n"
         << "  func.func @main(%x: " << t_in << ", %wg: " << t_wg << ", %wu: " << t_wg << ", %wd: " << t_wd << ") -> " << t_out << " {\n"
         << "    %g = stablehlo.dot_general %x, %wg, contracting_dims = [1] x [1] : (" << t_in << ", " << t_wg << ") -> " << t_hf32 << "\n"
         << "    %u = stablehlo.dot_general %x, %wu, contracting_dims = [1] x [1] : (" << t_in << ", " << t_wg << ") -> " << t_hf32 << "\n"
         << "    %c_half = stablehlo.constant dense<5.000000e-01> : " << t_hf32 << "\n"
         << "    %c_one  = stablehlo.constant dense<1.000000e+00> : " << t_hf32 << "\n"
         << "    %c_s2p  = stablehlo.constant dense<7.97884583e-01> : " << t_hf32 << "\n"
         << "    %c_coef = stablehlo.constant dense<4.471500e-02> : " << t_hf32 << "\n"
         << "    %g2 = stablehlo.multiply %g, %g : " << t_hf32 << "\n"
         << "    %g3 = stablehlo.multiply %g2, %g : " << t_hf32 << "\n"
         << "    %cg3 = stablehlo.multiply %c_coef, %g3 : " << t_hf32 << "\n"
         << "    %inner = stablehlo.add %g, %cg3 : " << t_hf32 << "\n"
         << "    %scaled = stablehlo.multiply %c_s2p, %inner : " << t_hf32 << "\n"
         << "    %th = stablehlo.tanh %scaled : " << t_hf32 << "\n"
         << "    %one_p = stablehlo.add %c_one, %th : " << t_hf32 << "\n"
         << "    %cdf = stablehlo.multiply %c_half, %one_p : " << t_hf32 << "\n"
         << "    %gelu = stablehlo.multiply %g, %cdf : " << t_hf32 << "\n"
         << "    %act_f32 = stablehlo.multiply %gelu, %u : " << t_hf32 << "\n"
         << "    %act_bf16 = stablehlo.convert %act_f32 : (" << t_hf32 << ") -> " << t_hbf << "\n"
         << "    %out = stablehlo.dot_general %act_bf16, %wd, contracting_dims = [1] x [1] : (" << t_hbf << ", " << t_wd << ") -> " << t_out << "\n"
         << "    return %out : " << t_out << "\n"
         << "  }\n"
         << "}\n";

    // Compile portable executable once; runs on any of the addressable TPU chips!
    PJRT_LoadedExecutable* exec = compile_stablehlo(mlir.str(), -1);
    if (!exec) return nullptr;

    std::lock_guard<std::mutex> lock(cache_mutex_);
    exec_cache_[key] = exec;
    return exec;
}

bool HavenTpuPjrtBackend::matmul_resident_nt(const float* x_host, int batch_m,
                                             const TpuMatrixHandle& w_resident,
                                             float* y_host) {
    if (!initialized_ || !x_host || !y_host || !w_resident.buffer || batch_m <= 0) return false;
    const int N = static_cast<int>(w_resident.rows);
    const int K = static_cast<int>(w_resident.cols);
    PJRT_LoadedExecutable* exec = get_or_compile_matmul_nt(batch_m, N, K, w_resident.device_index);
    if (!exec) return false;

    std::vector<uint16_t> x_bf16(static_cast<size_t>(batch_m) * static_cast<size_t>(K));
    convert_f32_to_bf16(x_host, x_bf16.data(), x_bf16.size());
    PJRT_Buffer* x_dev = upload_bf16_2d(x_bf16.data(), batch_m, K, w_resident.device_index);
    if (!x_dev) return false;

    std::vector<PJRT_Buffer*> outputs;
    bool ok = execute_1d(exec, w_resident.device_index, {x_dev, w_resident.buffer}, 1, outputs);
    destroy_buffer(x_dev);
    if (!ok || outputs.empty() || !outputs[0]) return false;

    ok = download_f32(outputs[0], y_host, static_cast<size_t>(batch_m) * static_cast<size_t>(N));
    destroy_buffer(outputs[0]);
    return ok;
}

bool HavenTpuPjrtBackend::fused_geglu_ffn(const float* x_host, int batch_m,
                                          const TpuMatrixHandle& w_gate,
                                          const TpuMatrixHandle& w_up,
                                          const TpuMatrixHandle& w_down,
                                          float* y_host) {
    if (!initialized_ || !x_host || !y_host || !w_gate.buffer || !w_up.buffer || !w_down.buffer || batch_m <= 0) {
        return false;
    }
    const int hidden_dim = static_cast<int>(w_gate.rows);
    const int emb_dim = static_cast<int>(w_gate.cols);
    if (w_up.rows != hidden_dim || w_up.cols != emb_dim || w_down.rows != emb_dim || w_down.cols != hidden_dim) {
        last_error_ = "Fused GeGLU dimension mismatch.";
        return false;
    }
    PJRT_LoadedExecutable* exec = get_or_compile_geglu_ffn(batch_m, hidden_dim, emb_dim, w_gate.device_index);
    if (!exec) return false;

    std::vector<uint16_t> x_bf16(static_cast<size_t>(batch_m) * static_cast<size_t>(emb_dim));
    convert_f32_to_bf16(x_host, x_bf16.data(), x_bf16.size());
    PJRT_Buffer* x_dev = upload_bf16_2d(x_bf16.data(), batch_m, emb_dim, w_gate.device_index);
    if (!x_dev) return false;

    std::vector<PJRT_Buffer*> outputs;
    bool ok = execute_1d(exec, w_gate.device_index, {x_dev, w_gate.buffer, w_up.buffer, w_down.buffer}, 1, outputs);
    destroy_buffer(x_dev);
    if (!ok || outputs.empty() || !outputs[0]) return false;

    ok = download_f32(outputs[0], y_host, static_cast<size_t>(batch_m) * static_cast<size_t>(emb_dim));
    destroy_buffer(outputs[0]);
    return ok;
}

TpuGemmBenchResult HavenTpuPjrtBackend::benchmark_bf16_gemm(int M, int N, int K, int iterations, int num_devices) {
    TpuGemmBenchResult res;
    res.M = M; res.N = N; res.K = K;
    res.iterations = std::max(1, iterations);
    if (!initialized_ || devices_.empty()) {
        res.error = "PJRT TPU backend is not initialized.";
        return res;
    }

    const int use_devs = std::clamp(num_devices, 1, static_cast<int>(devices_.size()));
    res.num_devices = use_devs;

    // Compile one BF16 GEMM executable per active TPU TensorCore
    std::ostringstream mlir;
    mlir << "module @haven_tpu_gemm {\n"
         << "  func.func @main(%lhs: tensor<" << M << "x" << K << "xbf16>, "
         << "%rhs: tensor<" << K << "x" << N << "xbf16>) -> tensor<" << M << "x" << N << "xbf16> {\n"
         << "    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0] "
         << ": (tensor<" << M << "x" << K << "xbf16>, tensor<" << K << "x" << N << "xbf16>) "
         << "-> tensor<" << M << "x" << N << "xbf16>\n"
         << "    return %0 : tensor<" << M << "x" << N << "xbf16>\n"
         << "  }\n"
         << "}\n";
    const std::string mlir_str = mlir.str();

    auto t_comp0 = std::chrono::high_resolution_clock::now();
    // Compile ONE portable executable and launch in parallel across all active TPU TensorCores!
    PJRT_LoadedExecutable* portable_exec = compile_stablehlo(mlir_str, -1);
    if (!portable_exec) {
        res.error = last_error_;
        return res;
    }
    auto t_comp1 = std::chrono::high_resolution_clock::now();
    res.compile_ms = std::chrono::duration<double, std::milli>(t_comp1 - t_comp0).count();

    // Prepare deterministic BF16 inputs on host and transfer once to each chip's HBM3
    std::vector<uint16_t> host_a(static_cast<size_t>(M) * static_cast<size_t>(K));
    std::vector<uint16_t> host_b(static_cast<size_t>(K) * static_cast<size_t>(N));
    const uint16_t val_a = fp32_to_bf16(0.015625f); // 1/64 exact in BF16
    const uint16_t val_b = fp32_to_bf16(0.015625f);
    std::fill(host_a.begin(), host_a.end(), val_a);
    std::fill(host_b.begin(), host_b.end(), val_b);

    std::vector<PJRT_Buffer*> dev_a(static_cast<size_t>(use_devs), nullptr);
    std::vector<PJRT_Buffer*> dev_b(static_cast<size_t>(use_devs), nullptr);
    for (int d = 0; d < use_devs; ++d) {
        dev_a[static_cast<size_t>(d)] = upload_bf16_2d(host_a.data(), M, K, d);
        dev_b[static_cast<size_t>(d)] = upload_bf16_2d(host_b.data(), K, N, d);
        if (!dev_a[static_cast<size_t>(d)] || !dev_b[static_cast<size_t>(d)]) {
            res.error = last_error_;
            for (int k = 0; k < use_devs; ++k) {
                destroy_buffer(dev_a[static_cast<size_t>(k)]);
                destroy_buffer(dev_b[static_cast<size_t>(k)]);
            }
            destroy_executable(portable_exec);
            return res;
        }
    }

    auto run_wave = [&](bool verify_first) -> bool {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(use_devs));
        std::vector<bool> flags(static_cast<size_t>(use_devs), false);
        float sample_val = 0.0f;

        for (int d = 0; d < use_devs; ++d) {
            threads.emplace_back([&, d]() {
                std::vector<PJRT_Buffer*> outs;
                if (!execute_1d(portable_exec, d,
                                {dev_a[static_cast<size_t>(d)], dev_b[static_cast<size_t>(d)]}, 1, outs)) {
                    return;
                }
                if (verify_first && d == 0) {
                    std::vector<uint16_t> out_h(static_cast<size_t>(M) * static_cast<size_t>(N));
                    if (download_bf16(outs[0], out_h.data(), out_h.size())) {
                        sample_val = bf16_to_fp32(out_h[0]);
                    }
                }
                destroy_buffer(outs[0]);
                flags[static_cast<size_t>(d)] = true;
            });
        }
        for (auto& th : threads) th.join();
        if (verify_first) res.checksum = sample_val;
        for (bool f : flags) if (!f) return false;
        return true;
    };

    // Warmup wave (populates XLA/TPU HBM cache & checks numerical output = K * (1/64) * (1/64))
    if (!run_wave(true)) {
        res.error = last_error_.empty() ? "Warmup execution failed." : last_error_;
    } else {
        auto t0 = std::chrono::high_resolution_clock::now();
        bool all_ok = true;
        for (int it = 0; it < res.iterations; ++it) {
            if (!run_wave(false)) { all_ok = false; break; }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        if (all_ok) {
            const double total_s = std::chrono::duration<double>(t1 - t0).count();
            res.avg_step_ms = (total_s * 1000.0) / static_cast<double>(res.iterations);
            const double flops_per_core = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
            const double total_flops = flops_per_core * static_cast<double>(use_devs) * static_cast<double>(res.iterations);
            res.total_tflops = (total_flops / total_s) / 1e12;
            res.per_device_tflops = res.total_tflops / static_cast<double>(use_devs);
            res.ok = true;
        } else {
            res.error = last_error_.empty() ? "Timed execution failed." : last_error_;
        }
    }

    for (int d = 0; d < use_devs; ++d) {
        destroy_buffer(dev_a[static_cast<size_t>(d)]);
        destroy_buffer(dev_b[static_cast<size_t>(d)]);
    }
    destroy_executable(portable_exec);
    return res;
}

#endif // _WIN32

} // namespace haven
