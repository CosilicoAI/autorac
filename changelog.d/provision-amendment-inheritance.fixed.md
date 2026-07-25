Provision-level corpus targets now inherit their parent document's identifiers
for amendment discovery when both rows share a non-empty source path and version.
This preserves document-level behavior while preventing cross-document leakage.

Closes #1275.
