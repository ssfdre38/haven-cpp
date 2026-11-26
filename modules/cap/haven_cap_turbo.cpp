#include "haven/haven_module_abi.h"

namespace {

static const HavenModuleDescriptor kDesc = {
    HAVEN_MODULE_ABI_VERSION,
    HAVEN_MOD_CAPABILITY,
    "haven_cap_turbo",
    "Haven Gemma 4.5 Turbo Block-FWHT & Folded-Norm Capability",
    "1.0.0",
    "turbo,fwht,quarot",
    HAVEN_FEAT_TURBO_FWHT,
    100
};

} // namespace

HAVEN_MOD_EXPORT const HavenModuleDescriptor* haven_module_query_descriptor() {
    return &kDesc;
}
