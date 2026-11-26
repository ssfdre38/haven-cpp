#include "haven/haven_module_abi.h"

namespace {

static const HavenModuleDescriptor kDesc = {
    HAVEN_MODULE_ABI_VERSION,
    HAVEN_MOD_CAPABILITY,
    "haven_cap_moe",
    "Haven Sparse MoE Router & Expert Grouping Capability",
    "1.0.0",
    "moe,experts",
    HAVEN_FEAT_MOE,
    100
};

} // namespace

HAVEN_MOD_EXPORT const HavenModuleDescriptor* haven_module_query_descriptor() {
    return &kDesc;
}
