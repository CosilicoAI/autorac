Execute RuleSpec multiline `if` and `elif` condition continuations with the
pinned expression grammar so source-completeness evidence follows the same
reachable branch as the rules engine. The validator now handles nested and
inline conditionals with lazy selector evaluation, enforces the pinned scalar
envelope for numbers and dates, and keeps malformed or unsupported expression
trees and bare field access fail closed.
