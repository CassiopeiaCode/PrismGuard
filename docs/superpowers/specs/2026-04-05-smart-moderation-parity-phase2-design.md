# Smart Moderation Parity Phase 2 Design

**Scope**

Bring the Rust smart-moderation control flow closer to the Python reference in areas that still affect behavior, while staying inside `/services/apps/Prismguand-Rust` and preserving the current passing test baseline.

**What Is Already Aligned**

- Three local model families are available in Rust: `bow`, `fasttext`, `hashlinear`.
- Pebble/RocksDB history lookup and cleanup parity is covered by dedicated tests.
- Four-format conversion and proxy request/response/stream paths already have broad test coverage and currently pass in a single-core `--no-default-features` aggregate run.

**Remaining Behavior Gaps**

1. AI review sampling semantics differ.
Python uses `random.random()` on each request. Rust currently uses a deterministic hash of `(seed, text)`, which makes repeated requests for the same text always choose the same sampling path.

2. AI model retry selection differs.
Python chooses an untried model at random first, then allows repeats after the candidate set is exhausted. Rust currently rotates through the candidate list by attempt index.

3. Profile caching semantics differ.
Python caches `ModerationProfile` instances globally. Rust reloads from disk for every request. This is usually harmless, but it is still a control-plane difference and can change hot-reload vs consistency behavior.

4. LRU pool bounds differ.
Python caps the number of profile caches. Rust currently bounds entries per profile but not the number of profile buckets.

**Non-Goals**

- Rewriting the proxy format-conversion subsystem again.
- Reproducing Python logging output line-for-line.
- Reproducing Python's exact client-library stack for OpenAI calls.

**Recommended Approach**

Use a minimal parity patch set that changes only control-flow semantics with direct tests:

- Make AI review-rate sampling match Python's request-time random draw, while keeping the existing configurable seed available for deterministic tests where needed.
- Match Python's "prefer untried models first" retry policy without changing the external HTTP contract.
- Add optional in-process profile caching and bounded cache-bucket eviction to mirror Python's cache structure.

This is the best tradeoff because it closes real behavior gaps without destabilizing the already-green request/response conversion stack.

**Design**

## 1. Smart moderation decision parity

`src/moderation/smart.rs` should keep the current three-stage decision flow, but the "force AI review" branch should use a per-request random draw instead of a text-hash draw. This is the largest remaining semantic mismatch with Python because it changes whether duplicate clean texts can still be sampled into AI labeling traffic.

The implementation should isolate the policy in a small helper so tests can verify edge cases such as `0.0`, `1.0`, and intermediate probabilities. The helper should not change the cache behavior: if the result is already cached, the cache still short-circuits before AI or local-model routing, just as it does today.

## 2. AI retry model selection parity

The retry loop in `src/moderation/smart.rs` should be refactored so model selection is policy-driven. The policy should first choose among not-yet-attempted models, and only when all candidates were used should it allow a repeat. For testability and single-core reproducibility, the implementation can use a seeded RNG helper rather than ambient global randomness.

This does not need to mimic Python's OpenAI SDK internals. Only the candidate-selection semantics matter.

## 3. Profile/cache parity

`src/profile.rs` should expose a lightweight cached profile loader for request-time paths, while preserving the existing plain `load()` entry point for tests and explicit reload cases. `src/moderation/smart.rs` should use the cached loader.

The moderation result cache should also bound the number of profile buckets, similar to Python's `MAX_PROFILES`, so profile churn cannot grow unbounded even if each individual bucket remains LRU-bounded.

**Validation**

- Add focused unit tests for AI review-rate sampling policy.
- Add focused unit tests for model candidate selection order and no-repeat-before-exhaustion semantics.
- Add focused unit tests for bounded profile-cache bucket eviction if that change is implemented.
- Re-run the existing single-core aggregate integration suite after the patch set.

**Expected Outcome**

After this phase, the Rust smart-moderation control path should match Python more closely in request-time routing decisions and retry behavior, while keeping the currently passing proxy and storage parity suites intact.
