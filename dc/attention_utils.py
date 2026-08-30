import math
from typing import Callable, List, Optional, Tuple

import torch


class STGIdentityValueAttnProcessor:
    """Paper-faithful STG-A processor.

    The STG paper describes attention skip by replacing the attention matrix
    A with the identity I so that SA'(Q, K, V) = IV = V. This processor
    implements that directly in diffusers' attention path by skipping the
    query/key attention-score computation while preserving the value and output
    projections.
    """

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)
        hidden_states = attn.batch_to_head_dim(value)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states


class CrossAttentionMapRecorder:
    """Collect token-conditioned cross-attention maps from selected UNet layers."""

    def __init__(
        self,
        token_indices: List[int],
        conditioned_batch_size: int,
        reference_spatial_shape: Optional[Tuple[int, int]] = None,
    ):
        self.token_indices = sorted(set(token_indices))
        self.conditioned_batch_size = conditioned_batch_size
        self.reference_spatial_shape = reference_spatial_shape
        self.maps: List[torch.Tensor] = []

    def _infer_spatial_shape(self, query_tokens: int) -> Optional[Tuple[int, int]]:
        """Infer a non-square spatial shape using the latent aspect ratio."""
        if self.reference_spatial_shape is None:
            side = int(round(math.sqrt(query_tokens)))
            if side * side != query_tokens:
                return None
            return (side, side)

        ref_h, ref_w = self.reference_spatial_shape
        if ref_h <= 0 or ref_w <= 0:
            return None

        target_ratio = ref_w / ref_h
        best_shape = None
        best_error = None

        # Iterate over all divisor pairs in BOTH orientations so that portrait latents (ref_h > ref_w) are matchable.
        for divisor in range(1, int(math.sqrt(query_tokens)) + 1):
            if query_tokens % divisor != 0:
                continue
            other = query_tokens // divisor
            for (h, w) in ((divisor, other), (other, divisor)):
                ratio_error = abs(math.log((w / h) / target_ratio))
                if best_error is None or ratio_error < best_error:
                    best_error = ratio_error
                    best_shape = (h, w)

        return best_shape

    def record(
        self,
        attention_probs: torch.Tensor,
        num_heads: int,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        if not self.token_indices:
            return

        batch_times_heads, query_tokens, key_tokens = attention_probs.shape
        if num_heads <= 0 or batch_times_heads % num_heads != 0:
            return

        batch_size = batch_times_heads // num_heads
        if batch_size < self.conditioned_batch_size:
            return

        valid_token_indices = [idx for idx in self.token_indices if 0 <= idx < key_tokens]
        if not valid_token_indices:
            return

        attn = attention_probs.view(batch_size, num_heads, query_tokens, key_tokens)
        attn = attn[: self.conditioned_batch_size]
        token_map = attn[..., valid_token_indices].mean(dim=-1).mean(dim=1)

        if spatial_shape is None:
            spatial_shape = self._infer_spatial_shape(query_tokens)
            if spatial_shape is None:
                return

        token_map = token_map.view(self.conditioned_batch_size, 1, *spatial_shape)
        self.maps.append(token_map.detach())


class CrossAttentionCaptureProcessor:
    """Default attention processor plus token-map recording for cross-attention."""

    def __init__(self, recorder: CrossAttentionMapRecorder):
        self.recorder = recorder

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        spatial_shape = None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            spatial_shape = (height, width)
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if hasattr(attn, "prepare_attention_mask"):
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.recorder.record(attention_probs, getattr(attn, "heads", 1), spatial_shape=spatial_shape)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, *spatial_shape)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states


def run_unet_with_skipped_attn(unet, device, skip_layers, latent_model_input, t, text_embeddings):
    """Run UNet forward pass with STG-A attention skip in selected up_blocks."""
    hooks = []
    original_processors = []
    stg_processor = STGIdentityValueAttnProcessor()

    def _zero_attn_branch_output(module, inputs, output):
        return torch.zeros_like(output)

    try:
        for layer_idx in skip_layers:
            block = unet.up_blocks[layer_idx]
            if not hasattr(block, "attentions"):
                continue
            for attn_module in block.attentions:
                for transformer_block in attn_module.transformer_blocks:
                    attn = transformer_block.attn1
                    if hasattr(attn, "processor"):
                        original_processors.append((attn, attn.processor))
                        if hasattr(attn, "set_processor"):
                            attn.set_processor(stg_processor)
                        else:
                            attn.processor = stg_processor
                    else:
                        hook = attn.register_forward_hook(_zero_attn_branch_output)
                        hooks.append(hook)

        with torch.no_grad():
            output = unet.forward(
                latent_model_input,
                torch.cat([t] * 3).to(device),
                encoder_hidden_states=text_embeddings,
            )
        return output.sample
    finally:
        for attn, processor in original_processors:
            if hasattr(attn, "set_processor"):
                attn.set_processor(processor)
            else:
                attn.processor = processor
        for hook in hooks:
            hook.remove()


def run_unet_with_cross_attention_capture(
    unet,
    cross_attention_layers,
    token_indices,
    latent_model_input,
    t,
    text_embeddings,
    conditioned_batch_size,
    build_attention_mask_fn: Callable[[List[torch.Tensor]], Optional[torch.Tensor]],
):
    """Run the UNet once while recording target-token cross-attention maps."""
    if not token_indices:
        output = unet.forward(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
        )
        return output.sample, None

    recorder = CrossAttentionMapRecorder(
        token_indices=token_indices,
        conditioned_batch_size=conditioned_batch_size,
        reference_spatial_shape=latent_model_input.shape[-2:],
    )
    capture_processor = CrossAttentionCaptureProcessor(recorder)
    original_processors = []

    try:
        for layer_idx in cross_attention_layers:
            if layer_idx < 0 or layer_idx >= len(unet.up_blocks):
                continue
            block = unet.up_blocks[layer_idx]
            if not hasattr(block, "attentions"):
                continue
            for attn_module in block.attentions:
                for transformer_block in attn_module.transformer_blocks:
                    attn = transformer_block.attn2
                    if hasattr(attn, "processor"):
                        original_processors.append((attn, attn.processor))
                        if hasattr(attn, "set_processor"):
                            attn.set_processor(capture_processor)
                        else:
                            attn.processor = capture_processor

        output = unet.forward(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
        )
        attention_mask = build_attention_mask_fn(recorder.maps)
        return output.sample, attention_mask
    finally:
        for attn, processor in original_processors:
            if hasattr(attn, "set_processor"):
                attn.set_processor(processor)
            else:
                attn.processor = processor
