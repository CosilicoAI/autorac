Persisted-row revalidation now normalizes outer and inner harness temporary
paths, subprocess capture paths, and path-bearing capture errors, and
deterministically orders validation issues, including compile-timeout commands.
Staging roots normalize at quotes, commas, parentheses, and other punctuation
boundaries without rewriting longer path-prefix tokens, so repeated validation
of one artifact produces identical metrics.
