# Contributing to Haven C++ (`haven-cpp`)

Thank you for your interest in contributing to `haven-cpp`.

## Dual-Licensing & Contributor License Agreement (CLA)

Because `haven-cpp` is dual-licensed under the **GNU Affero General Public License v3.0 (AGPLv3)** and the **Haven-CPP Commercial License**, the project maintainer (Daniel Elliott) must hold sufficient rights to distribute all contributions under both open-source and commercial licenses.

By submitting a pull request, patch, issue attachment, or other contribution to this repository, you certify and agree that:

1. **Originality & Authority:** Your contribution is your original creation (or you have full legal authority from your employer/rights holder to submit it under these terms).
2. **Perpetual Dual-Licensing Grant:** You grant **Daniel Elliott** (and successors/assigns) a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright and patent license to reproduce, prepare derivative works of, publicly display, publicly perform, sublicense, and distribute your contribution under **both** the GNU AGPLv3 and the Haven-CPP Commercial License (or any future open-source or commercial license chosen for `haven-cpp`).
3. **No Obligation of Inclusion:** Submission does not obligate the maintainer to merge or include the contribution.

If you do not wish to grant these rights, please do not submit contributions to this repository.

## Development & Verification Standards

Before opening a pull request:

1. Ensure the build compiles cleanly in C++20 (`Release`) with zero new warnings.
2. Run the offline StableHLO IR verifier and full adversarial security suite:
   ```bash
   ./build/test-tpu-gemma4-onchip --emit-only <model.gguf>
   ./build/test-security-fuzz
   ./build/test-secure-cluster
   ```
3. Never introduce third-party code with incompatible licenses without explicit maintainer approval and attribution in [NOTICE](NOTICE).
