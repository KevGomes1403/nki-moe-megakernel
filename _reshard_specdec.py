"""One-off: regenerate the presharded checkpoint for an already-compiled build.

The Stage C build's HLO/NEFF compile (model.pt) completed, but the process was
killed during the post-compile weight-sharding step, leaving weights/ empty.
This re-runs ONLY the CPU-side shard+serialize (shard_weights) -- no recompile.

Run from repo root:
    A3B_ENABLE_MTP=1 A3B_ENABLE_VERIFY=1 A3B_RETURN_LOGITS=1 \
    python -m models.qwen3_6_moe._reshard_specdec
"""

import os

os.environ.setdefault("A3B_ENABLE_MTP", "1")
os.environ.setdefault("A3B_ENABLE_VERIFY", "1")
os.environ.setdefault("A3B_RETURN_LOGITS", "1")

from models.qwen3_6_moe.inference_qwen36_a3b import build_inference_config
from models.qwen3_6_moe.modeling_qwen36_a3b import NeuronQwen36A3BForCausalLM

MODEL = "/home/ubuntu/models/Qwen3.6-35B-A3B"
COMPILED = os.environ.get(
    "A3B_COMPILED_PATH", "/home/ubuntu/models/qwen36_a3b_specdec_traced"
)
TP = 4
SEQ = 128


def main():
    cfg = build_inference_config(MODEL, TP, SEQ)
    model = NeuronQwen36A3BForCausalLM(MODEL, cfg)
    print("Sharding weights (no recompile)...")
    model.shard_weights(COMPILED)
    print("Done. weights/ should now contain tp{0..3}_sharded_checkpoint.safetensors")


if __name__ == "__main__":
    main()
