import re
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


DEFAULT_CROSS_ATTENTION_STOPWORDS = {
    "a", "an", "the", "of", "to", "into", "in", "on", "at", "for", "with", "from",
    "photo", "image", "picture", "turn", "make", "him", "her", "them", "his", "their",
    "this", "that", "is", "as", "be", "and", "or",
}


def normalize_relevance_map(relevance: torch.Tensor) -> torch.Tensor:
    """Percentile-normalize a relevance map to [0, 1] per sample (p5–p95 window)."""
    flat = relevance.flatten(1)
    p5 = torch.quantile(flat, 0.05, dim=1, keepdim=True).view(-1, 1, 1, 1)
    p95 = torch.quantile(flat, 0.95, dim=1, keepdim=True).view(-1, 1, 1, 1)
    return ((relevance - p5) / (p95 - p5 + 1e-8)).clamp(0.0, 1.0)


def apply_mask_postprocessing(mask: torch.Tensor, gamma: float, sigma: float) -> torch.Tensor:
    """Sharpen a mask with a gamma power curve, then soften edges with a Gaussian blur."""
    if gamma != 1.0:
        mask = mask.clamp_min(0.0).pow(gamma)

    if sigma > 0:
        kernel_size = max(3, int(round(6 * sigma + 1)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = TF.gaussian_blur(mask, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    return mask.clamp(0.0, 1.0)


def derive_cross_attention_keywords(src_prompt: Optional[str], tgt_prompt: str) -> List[str]:
    """Auto-derive CA keywords: target-only words minus stopwords and source words.

    Kept prompt-based to remain fully general — no manual per-scene keyword overrides.
    """
    source_words = set(re.findall(r"[a-zA-Z0-9]+", (src_prompt or "").lower()))
    target_words = re.findall(r"[a-zA-Z0-9]+", tgt_prompt.lower())
    keywords = []
    for word in target_words:
        if word in DEFAULT_CROSS_ATTENTION_STOPWORDS or word in source_words:
            continue
        if word not in keywords:
            keywords.append(word)
    return keywords


def get_phrase_token_ids(tokenizer, phrase: str) -> List[int]:
    """Tokenize a phrase and return its token ids with special tokens stripped."""
    token_ids = tokenizer(
        phrase,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].tolist()
    special_ids = {
        token_id
        for token_id in (
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "cls_token_id", None),
            getattr(tokenizer, "sep_token_id", None),
        )
        if token_id is not None
    }
    return [token_id for token_id in token_ids if token_id not in special_ids]


def find_token_positions(full_token_ids: List[int], phrase_token_ids: List[int]) -> List[int]:
    """Return every position of `phrase_token_ids` as a contiguous subsequence of `full_token_ids`."""
    if not phrase_token_ids or len(phrase_token_ids) > len(full_token_ids):
        return []

    positions = []
    span = len(phrase_token_ids)
    for start in range(len(full_token_ids) - span + 1):
        if full_token_ids[start : start + span] == phrase_token_ids:
            positions.extend(range(start, start + span))
    return positions


def get_cross_attention_token_indices(
    tokenizer,
    tgt_prompt: str,
    src_prompt: Optional[str] = None,
) -> List[int]:
    """Locate the target prompt's edit-keyword token positions for CA-map capture.

    Falls back to all non-special token positions when no keyword survives
    stopword/source filtering.
    """
    full_token_ids = tokenizer(
        tgt_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0].tolist()

    keywords = derive_cross_attention_keywords(src_prompt, tgt_prompt)
    token_indices = []
    for keyword in keywords:
        token_indices.extend(find_token_positions(full_token_ids, get_phrase_token_ids(tokenizer, keyword)))

    if token_indices:
        return sorted(set(token_indices))

    special_ids = {
        token_id
        for token_id in (
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "cls_token_id", None),
            getattr(tokenizer, "sep_token_id", None),
        )
        if token_id is not None
    }
    return [idx for idx, token_id in enumerate(full_token_ids) if token_id not in special_ids]


def build_cross_attention_relevance_mask(
    attention_maps: List[torch.Tensor],
    gamma: float,
    sigma: float,
    target_shape=None,
):
    """Average captured cross-attention maps into a single normalized, postprocessed mask."""
    if not attention_maps:
        return None

    target_h = max(attn_map.shape[-2] for attn_map in attention_maps)
    target_w = max(attn_map.shape[-1] for attn_map in attention_maps)
    resized_maps = []
    for attn_map in attention_maps:
        if attn_map.shape[-2:] != (target_h, target_w):
            attn_map = F.interpolate(attn_map, size=(target_h, target_w), mode="bilinear", align_corners=False)
        resized_maps.append(attn_map)

    relevance = torch.stack(resized_maps, dim=0).mean(dim=0)
    normalized = normalize_relevance_map(relevance)
    mask = apply_mask_postprocessing(
        normalized,
        gamma=gamma,
        sigma=sigma,
    )
    if target_shape is not None and mask.shape[-2:] != tuple(target_shape):
        mask = F.interpolate(mask, size=target_shape, mode="bilinear", align_corners=False)
    return mask.detach()
