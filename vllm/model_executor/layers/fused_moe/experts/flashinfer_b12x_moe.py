# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
import weakref
from dataclasses import dataclass
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.b12x import B12xWarmupUnit
from vllm.utils.flashinfer import (
    flashinfer_convert_sf_to_mma_layout,
    has_flashinfer_b12x_moe,
)

logger = init_logger(__name__)


@dataclass
class _B12xSharedCudaGraphBuffers:
    """B12x buffers reused by identically-shaped, sequential MoE layers."""

    static_workspace: Any
    dynamic_workspace: Any
    output: torch.Tensor

# vLLM invokes these same-shaped layers sequentially and copies each wrapper's
# result into caller-owned output before invoking the next layer, so their large
# graph-safe workspaces and staging output do not need per-layer allocations.
# A model keeps each entry alive through ``FlashInferB12xExperts._shared_buffers``.
# Weak values avoid retaining GPU memory after the model is unloaded.  Access is
# serialized because model construction can prepare layers from multiple threads.
_B12X_SHARED_BUFFERS: weakref.WeakValueDictionary[
    tuple[Any, ...], _B12xSharedCudaGraphBuffers
] = weakref.WeakValueDictionary()
_B12X_SHARED_BUFFERS_LOCK = threading.Lock()


@dataclass
class _B12xSharedCudaGraphBuffers:
    """B12x buffers reused by identically-shaped, sequential MoE layers."""

    static_workspace: Any
    dynamic_workspace: Any
    output: torch.Tensor

# vLLM invokes these same-shaped layers sequentially and copies each wrapper's
# result into caller-owned output before invoking the next layer, so their large
# graph-safe workspaces and staging output do not need per-layer allocations.
# A model keeps each entry alive through ``FlashInferB12xExperts._shared_buffers``.
# Weak values avoid retaining GPU memory after the model is unloaded.  Access is
# serialized because model construction can prepare layers from multiple threads.
_B12X_SHARED_BUFFERS: weakref.WeakValueDictionary[
    tuple[Any, ...], _B12xSharedCudaGraphBuffers
] = weakref.WeakValueDictionary()
_B12X_SHARED_BUFFERS_LOCK = threading.Lock()


class FlashInferB12xExperts(mk.FusedMoEExpertsModular):
    """FlashInfer CuteDSL fused MoE expert for SM12x (SM120/SM121,
    RTX Pro 6000 / DGX Spark).

    Uses ``b12x_fused_moe`` from FlashInfer PR #3080 which fuses token
    dispatch, two GEMMs, SwiGLU activation, and topk-weight reduction into a
    single kernel call.  Input quantization (BF16→FP4) is performed inside the
    kernel so BF16 hidden states are passed directly.

    Weight scale factors are converted to the MMA layout produced by
    ``convert_sf_to_mma_layout`` once during ``process_weights_after_loading``
    and cached as ``w1_sf_mma`` / ``w2_sf_mma``.

    Only NVFP4 (kNvfp4Static/kNvfp4Dynamic) quantization is supported.
    """

    _ACTIVATION_MAP: dict[MoEActivation, str] = {
        MoEActivation.SILU: "silu",
        MoEActivation.GELU_TANH: "gelu_tanh",
        MoEActivation.RELU2_NO_MUL: "relu2",
    }

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.quant_dtype == "nvfp4", (
            "FlashInferB12xExperts only supports nvfp4 quantization."
        )
        self.out_dtype = moe_config.in_dtype
        self.num_local_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        # FC2 input scale tensor bound in process_weights_after_loading: the
        # calibrated (now-zeroed) a2_gscale for static-quant checkpoints, or
        # a synthesized uniform-1.0 tensor for W4A16 checkpoints that lack
        # one. Holding it on the instance keeps apply() alloc-free.
        self._fc2_input_scale: torch.Tensor | None = None

        # Shape params for B12xMoEWrapper construction.
        self.global_num_experts = moe_config.num_experts
        self.topk = moe_config.experts_per_token
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.max_num_tokens = moe_config.max_num_tokens
        self.local_expert_offset = self.ep_rank * self.num_local_experts

        activation = moe_config.activation
        if activation not in self._ACTIVATION_MAP:
            raise ValueError(
                f"FlashInferB12xExperts does not support "
                f"activation {activation!r}. "
                f"Supported: {list(self._ACTIVATION_MAP.keys())}"
            )
        self._activation_str = self._ACTIVATION_MAP[activation]

        # Lazily created on first apply() call.
        self._wrapper: Any | None = None
        self._shared_buffers: _B12xSharedCudaGraphBuffers | None = None
        self.w1_sf_mma: torch.Tensor | None = None
        self.w2_sf_mma: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Normalise block scales to absorb the per-expert weight global scale
        # (w_gs).  vLLM's NVFP4 convention stores:
        #   block_scale = max_abs * w_gs / fp4_max,  g1_alphas = 1/w_gs
        # The SM12x kernel treats w1_alpha (= g1_alphas) as a per-expert weight
        # dequant multiplier separate from input_gs (activation scale).  We bake
        # w_gs into the block scales so that w1_alpha = 1.0 and the kernel sees
        # the simpler form:
        #   block_scale = max_abs / fp4_max,  w1_alpha = 1.0
        # The FP4-packed values and dequantised results are identical in both
        # representations.  We set scale_2 = 1.0 to signal that the bake-in is
        # already done.
        layer.w13_weight_scale.data = (
            layer.w13_weight_scale.float() * layer.w13_weight_scale_2.view(-1, 1, 1)
        ).to(layer.w13_weight_scale.dtype)
        layer.w13_weight_scale_2.data.fill_(1.0)

        layer.w2_weight_scale.data = (
            layer.w2_weight_scale.float() * layer.w2_weight_scale_2.view(-1, 1, 1)
        ).to(layer.w2_weight_scale.dtype)
        layer.w2_weight_scale_2.data.fill_(1.0)

        # The SM12x kernel uses dynamic per-block quantization for FC2 input
        # activations (the SwiGLU output before the down projection).  The
        # calibrated a2_gscale from the modelopt checkpoint (~tens to hundreds)
        # is intended for static-quantisation backends (TRTLLM/CUTLASS) and
        # causes every intermediate activation to saturate at max FP4 when
        # multiplied by values that large.  Force to 1.0 so the kernel uses
        # its own per-block dynamic scale.
        if self.a2_gscale is not None:
            self.a2_gscale.fill_(1.0)
            self._fc2_input_scale = self.a2_gscale
        else:
            # W4A16 NVFP4 checkpoints have no calibrated a2_gscale; b12x
            # performs dynamic per-block FC2-input quantization, so a uniform
            # 1.0 scale per expert is equivalent to the bake-in above for
            # static-quant checkpoints. Allocate once here so apply() stays
            # alloc-free.
            self._fc2_input_scale = torch.ones(
                self.num_local_experts,
                device=layer.w13_weight.device,
                dtype=torch.float32,
            )

        # Precompute MMA-layout views of the weight scale factors once here
        # rather than recomputing on every forward pass.
        assert self.w1_scale is not None
        num_experts_w1, m1, k1_sf = self.w1_scale.shape
        k1 = k1_sf * 16
        self.w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w1_scale.reshape(num_experts_w1 * m1, k1_sf),
            m=m1,
            k=k1,
            num_groups=num_experts_w1,
        )

        assert self.w2_scale is not None
        num_experts_w2, m2, k2_sf = self.w2_scale.shape
        k2 = k2_sf * 16
        self.w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w2_scale.reshape(num_experts_w2 * m2, k2_sf),
            m=m2,
            k=k2,
            num_groups=num_experts_w2,
        )

        # Let vllm.model_executor.warmup.b12x_warmup pre-compile this layer's
        # CuTe-DSL kernels and build any padded weight copies ahead of graph
        # capture instead of inside the capture window.
        layer.b12x_warmup_provider = self

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and has_flashinfer_b12x_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # b12x performs in-kernel BF16->FP4 activation quant, so W4A16
        # NVFP4 checkpoints (activation_key=None, e.g. mixed-precision
        # compressed-tensors layouts) are runtime-compatible.
        return (weight_key, activation_key) in (
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.GELU_TANH,
            MoEActivation.RELU2_NO_MUL,
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # B12xMoEWrapper does not yet support expert parallelism: its local
        # expert count must equal the global expert count.
        return not moe_parallel_config.use_ep

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # b12x_fused_moe applies topk weights internally.
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # b12x_fused_moe manages its own internal workspace.
        workspace1 = (1,)
        workspace2 = (0,)
        output_shape = (M, K)
        return (workspace1, workspace2, output_shape)

    @property
    def expects_unquantized_inputs(self) -> bool:
        # B12xMoEWrapper expects BF16 hidden states and performs its own FP4
        # quantization internally.  Returning True prevents the modular kernel
        # from pre-quantizing activations.
        return True

    def _avoid_static_repad(self) -> None:
        """Keep b12x's static backend from re-padding the expert weights.

        ``launch_sm120_moe`` pads the intermediate dim up to
        ``_STATIC_RETAINED_GROUP_N`` (256) when the static backend runs and ``n``
        is not already a multiple of it. The padded copy spans every expert, so
        for a 512-expert layer with n=640 (padded to 768) it is ~1 GiB per
        layer, built on the first static call and cached for the process. When
        that first call happens inside CUDA graph capture, the allocation lands
        in the capture window and can exhaust device memory.

        Sizes that are a multiple of the N tile (128) but not of the retained
        group (256) would pay that cost, so route them to the dynamic backend,
        which uses native N128 geometry and never pads. Shapes that are not
        N128-aligned already go through the wrapper's own pad-to-128 path and
        are unaffected.
        """
        n = self.intermediate_size_per_partition
        if n % 128 != 0 or n % 256 == 0:
            return
        if self.num_local_experts != self.global_num_experts:
            # The dynamic backend indexes global topk ids, so it requires
            # num_local_experts == num_experts. Leave EP alone.
            return
        var = "FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS"
        if var in os.environ:
            return
        os.environ[var] = "0"
        # The selector memoizes the cutover; drop any value cached before this
        # point so the override takes effect.
        try:
            import sys

            mod = sys.modules.get(
                "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"
            )
            if mod is not None:
                mod._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()
        except Exception:
            pass
        logger.info_once(
            "b12x MoE: intermediate_size=%d is not a multiple of 256, so the "
            "static backend would build a padded weight copy for every expert. "
            "Routing to the dynamic backend (%s=0) to avoid that allocation.",
            n,
            var,
        )

    def _ensure_wrapper(self) -> None:
        """Lazily create B12xMoEWrapper on first use."""
        if self._wrapper is not None:
            return

        self._avoid_static_repad()

        from flashinfer.fused_moe import B12xMoEWrapper

        device_index = torch.cuda.current_device()
        key = (
            device_index,
            self.global_num_experts,
            self.topk,
            self.hidden_dim,
            self.intermediate_size_per_partition,
            self.max_num_tokens,
            self.num_local_experts,
            self.out_dtype,
            self._activation_str,
        )
        wrapper_kwargs = dict(
            num_experts=self.global_num_experts,
            top_k=self.topk,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size_per_partition,
            use_cuda_graph=True,
            max_num_tokens=self.max_num_tokens,
            num_local_experts=self.num_local_experts,
            output_dtype=self.out_dtype,
            device=f"cuda:{device_index}",
            activation=self._activation_str,
        )

        with _B12X_SHARED_BUFFERS_LOCK:
            shared = _B12X_SHARED_BUFFERS.get(key)
            if shared is None:
                wrapper = B12xMoEWrapper(**wrapper_kwargs)
                shared = _B12xSharedCudaGraphBuffers(
                    static_workspace=wrapper._static_workspace,
                    dynamic_workspace=wrapper._dynamic_workspace,
                    output=wrapper._moe_output,
                )
                _B12X_SHARED_BUFFERS[key] = shared
            else:
                wrapper = B12xMoEWrapper(
                    **wrapper_kwargs,
                    shared_static_workspace=shared.static_workspace,
                    shared_dynamic_workspace=shared.dynamic_workspace,
                    shared_output=shared.output,
                )

            # Keep the weak-cache entry alive exactly as long as at least one
            # layer using the shared buffers remains alive.
            self._shared_buffers = shared
            self._wrapper = wrapper

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        """Compile this layer's serving-shape CuTe-DSL kernels ahead of capture."""
        assert output_dtype == self.out_dtype, (
            f"warmup dtype {output_dtype} != expert dtype {self.out_dtype}"
        )
        device = torch.device(layer.w13_weight.device)

        def compile() -> None:
            for tokens in token_counts:
                if tokens <= 0:
                    continue
                self.apply(
                    output=torch.empty(
                        (tokens, self.hidden_dim),
                        dtype=output_dtype,
                        device=device,
                    ),
                    hidden_states=torch.zeros(
                        (tokens, self.hidden_dim),
                        dtype=output_dtype,
                        device=device,
                    ),
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    topk_weights=torch.full(
                        (tokens, self.topk),
                        1.0 / self.topk,
                        dtype=torch.float32,
                        device=device,
                    ),
                    topk_ids=torch.zeros(
                        (tokens, self.topk),
                        dtype=torch.int32,
                        device=device,
                    ),
                    activation=layer.activation,
                    global_num_experts=self.global_num_experts,
                    expert_map=None,
                    a1q_scale=None,
                    a2_scale=None,
                    workspace13=None,
                    workspace2=None,
                    expert_tokens_meta=None,
                    apply_router_weight_on_input=bool(
                        getattr(layer, "apply_router_weight_on_input", False)
                    ),
                )

        return B12xWarmupUnit(
            name="MoE",
            key=(
                type(self),
                device,
                output_dtype,
                self.global_num_experts,
                self.num_local_experts,
                self.hidden_dim,
                self.intermediate_size_per_partition,
                self.topk,
                self.max_num_tokens,
                self._activation_str,
            ),
            compile=compile,
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for FlashInferB12xExperts"
        )
        assert self.g1_alphas is not None and self.g2_alphas is not None, (
            "g1_alphas and g2_alphas must not be None for FlashInferB12xExperts"
        )
        assert self._fc2_input_scale is not None, (
            "_fc2_input_scale must be set by process_weights_after_loading"
        )
        assert self.w1_sf_mma is not None and self.w2_sf_mma is not None, (
            "process_weights_after_loading must run before FlashInferB12xExperts.apply"
        )

        self._ensure_wrapper()
        wrapper = self._wrapper
        assert wrapper is not None

        # With VLLM_MOE_SKIP_PADDING enabled, the router represents padded
        # token routes as expert id -1 and gives them zero weight.  The B12x
        # kernels index expert metadata directly and do not accept the -1
        # sentinel, so map it to an in-range expert while preserving the
        # router's zero weight.  This is graph-capturable and makes padded
        # rows contribute exactly zero without an out-of-bounds expert read.
        safe_topk_ids = topk_ids.to(torch.int32).clamp_min(0)

        wrapper_output = wrapper.run(
            x=hidden_states,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self._fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.g2_alphas,
            token_selected_experts=safe_topk_ids,
            token_final_scales=topk_weights,
        )
        output.copy_(wrapper_output)
