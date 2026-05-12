"""Known-models registry.

Most models work without any per-model configuration — `encode_corpus(model=...)`
just downloads them from HF. The registry exists for the cases where a model
needs a special knob (e.g. the gte backbone ships custom transformer code on the
Hub, which `SparseEncoder` won't load without `trust_remote_code=True`).

Unknown models default to safe values; the registry is purely additive.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"


@dataclass(frozen=True)
class ModelSpec:
    doc_model_id: str
    trust_remote_code: bool = False
    # Pinned commit hash for the custom modeling code on HF. Used only when
    # trust_remote_code=True; pinning keeps us reproducible even if the model
    # repo silently changes its `modeling.py`.
    code_revision: str | None = None
    # When False, encode_corpus() raises before loading. Set with a `note`
    # explaining why so users get an actionable error.
    working: bool = True
    note: str | None = None


KNOWN_MODELS: dict[str, ModelSpec] = {
    # English, BERT-base backbone, ~67M params. Default.
    DEFAULT_MODEL: ModelSpec(doc_model_id=DEFAULT_MODEL),
    # English, GTE-base backbone (~137M). Ships custom transformer code on the Hub.
    # Requires the transformers<5 pin from pyproject.toml (the `encode` extra
    # pins it) plus our position_ids / weight-tying repairs in encoder.py.
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte": ModelSpec(
        doc_model_id="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
        trust_remote_code=True,
        code_revision="40ced75c3017eb27626c9d4ea981bde21a2662f4",
    ),
    # English, v2 distill — predecessor of v3-distill.
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill": ModelSpec(
        doc_model_id="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    ),
    # Multilingual; XLM-Roberta backbone.
    "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1": ModelSpec(
        doc_model_id="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    ),
}


def resolve(model_id: str) -> ModelSpec:
    """Look up the spec for a model; unknown models get safe defaults."""
    return KNOWN_MODELS.get(model_id, ModelSpec(doc_model_id=model_id))
