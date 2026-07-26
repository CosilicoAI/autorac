Persisted-row revalidation now normalizes outer and inner harness temporary
paths, subprocess capture paths, and path-bearing capture errors, and
deterministically orders validation issues, including compile-timeout commands,
so repeated validation of one artifact produces identical metrics.
