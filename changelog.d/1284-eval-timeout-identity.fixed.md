EncodeBench execution identity v3 now binds the effective Claude, Codex, and
OpenAI timeout policies plus generation retry limits. Terminal encoder timeout
evidence stops suite-level retries, bounding Claude and Codex timeout paths to
the two documented artifact-generation attempts instead of six.
