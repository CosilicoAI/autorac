Validator engine resolution now binds deterministically to the policy repo's
toolchain-pinned `axiom_rules_engine_ref` when one is declared: candidate
binaries must carry a binding receipt matching the pinned commit and the
binary's exact bytes, or the engine checkout must sit clean at the pin, in
which case the release profile is built on demand (`cargo build --release
--locked`) and receipted — instead of silently preferring whatever stale
`target/debug` build exists. Unpinned resolution now prefers release over
debug and logs an `engine_binding_unverified` event. `validate` and `compile`
accept `--axiom-rules-engine-ref` to override the toolchain pin, and the new
`engine-bind` subcommand verifies (or builds) and receipts a binary directly.

Deployment note: the strict corpus toolchain loader
(`load_rulespec_toolchain`) and the reviewed CI toolchain steps still require
the exact 3-key `[toolchain]` table, so declaring `axiom_rules_engine_ref` in
a strict repo's `.axiom/toolchain.toml` breaks corpus-bound commands and the
pinned "Resolve RuleSpec toolchain" CI step until that schema and the
workflow pins are widened in a coordinated bump. Until then, exercise the pin
through `--axiom-rules-engine-ref` on `validate`/`compile` or
`engine-bind --pin`.
