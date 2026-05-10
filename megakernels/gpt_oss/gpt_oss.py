# coding=utf-8
"""
GPT-OSS-20B XLA baseline for NXD inference.

Thin wrapper over NxDI's `NeuronGptOssForCausalLM` that pins compiler args to
match the style of the qwen baseline in this repo (saturate-infinity,
mixed-precision-accumulation, --lnc=2, vector-offset DGE). Apart from
`get_compiler_args`, behavior is unchanged from the upstream NxDI implementation.

Also pins reasonable defaults via a NeuronConfig subclass:
  - padded_hidden_size=3072 (next multiple of 256 from 2880; required for LNC-2)
  - padded_intermediate_size=3072
  - bf16 compute path (`is_mxfp4_compute=False`) — the megakernel sibling will
    keep the same default for v1 and switch to MXFP4 in a follow-up.
"""

import os
import shlex

# Pin trn3 platform target before any neuron import touches it.
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3")

from neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss import (  # noqa: E402
    GptOssInferenceConfig,
    GptOssNeuronConfig,
    NeuronGptOssForCausalLM as _BaseNeuronGptOssForCausalLM,
)
from neuronx_distributed_inference.models.model_wrapper import (  # noqa: E402
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)


class GptOssBaselineNeuronConfig(GptOssNeuronConfig):
    """GptOssNeuronConfig with hardware-friendly defaults for trn3 / LNC=2."""

    def __init__(self, **kwargs):
        # Hidden=2880, intermediate=2880. Both must be divisible by 256 (128*LNC=2)
        # for moe_block_tkg + LNC sharding. Pad to 3072 unless the user overrides.
        kwargs.setdefault("padded_hidden_size", 3072)
        kwargs.setdefault("padded_intermediate_size", 3072)
        # bf16 compute path; flip to True once the MXFP4 path is wired.
        kwargs.setdefault("is_mxfp4_compute", False)
        kwargs.setdefault("is_full_model_shuffled", False)
        kwargs.setdefault("sliding_window_attention_dp_degree", 1)
        # Strip the qwen-MoE-only kwarg that main.py injects unconditionally —
        # gpt-oss doesn't use BlockwiseMatmulConfig.
        kwargs.pop("blockwise_matmul_config", None)
        super().__init__(**kwargs)


class NeuronGptOssForCausalLM(_BaseNeuronGptOssForCausalLM):
    """GPT-OSS XLA baseline with custom compiler args matching this repo's style."""

    @classmethod
    def get_neuron_config_cls(cls):
        return GptOssBaselineNeuronConfig

    @classmethod
    def get_config_cls(cls):
        return GptOssInferenceConfig

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self):
        args = [
            "--enable-saturate-infinity",
            "--enable-mixed-precision-accumulation",
            "--model-type",
            "transformer",
            f"--lnc={self.neuron_config.logical_nc_config}",
        ]

        if getattr(self, "compile_tag", None) == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--enable-scalar-dge-vectorization",
            ]
            hlo2tensorizer_extra = "--modular-flow-mac-threshold=10"
        elif getattr(self, "compile_tag", None) == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O1"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--eager-tkg-vectorize-dma",
                "--enable-dge-on-indirect-dma",
                "--enable-dge-on-vector-indirect-dma",
            ]
            hlo2tensorizer_extra = ""
        else:
            optimization_level = "-O1"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--enable-scalar-dge-vectorization",
            ]
            hlo2tensorizer_extra = ""

        if tensorizer_opts:
            args.append(f"--tensorizer-options={' '.join(tensorizer_opts)}")
        args.append(optimization_level)
        args.append("--auto-cast=none")
        args += ["--internal-enable-dge-levels", "vector_dynamic_offsets"]

        if self.neuron_config.scratchpad_page_size:
            args.append(
                f"--hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size}"
            )

        if hlo2tensorizer_extra:
            args.append(
                f"--internal-hlo2tensorizer-options={hlo2tensorizer_extra} --verify-hlo=true"
            )

        return shlex.join(args)
