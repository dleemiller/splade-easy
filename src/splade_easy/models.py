"""Model defaults.

`DEFAULT_MODEL` is the doc encoder used when the caller doesn't supply one. Other
models can be passed via `encode_corpus(..., model="...")` and `SpladeRetriever(model="...")`;
their query-side IDF JSON is auto-discovered by `encoder._load_idf_map` walking a
few candidate paths in the HF repo, so no registry is required for them to work.
"""

from __future__ import annotations

DEFAULT_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
