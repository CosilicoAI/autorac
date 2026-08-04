Preserve the immediately preceding validator-rejected RuleSpec and companion
tests as bounded, untrusted edit context for the next encode attempt, so fixes
and historical cases accumulate across retries instead of being regenerated
from the legacy baseline. Retry capture now fails closed on unsafe paths,
symlinks, invalid UTF-8, or oversized candidates before deleting any output.
