#include "haven/haven_module_abi.h"

namespace {

static const HavenModuleDescriptor kDesc = {
    HAVEN_MODULE_ABI_VERSION,
    HAVEN_MOD_CAPABILITY,
    "haven_cap_vision",
    "Haven SigLIP Multimodal Vision Encoder Capability",
    "1.0.0",
    "vision,siglip,mmproj",
    HAVEN_FEAT_VISION_SIGLIP,
    100
};

} // namespace

HAVEN_MOD_EXPORT const HavenModuleDescriptor* haven_module_query_descriptor() {
    return &kDesc;
}
