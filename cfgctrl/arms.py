"""Named guidance laws shared by benchmark configuration and diagnostics."""
from dataclasses import replace
from typing import Dict, Tuple
from . import SMCConfig, presets

PRESETS = {
    "cfg": presets.cfg_baseline,
    "paper": presets.paper,
    "boundary_layer": presets.boundary_layer,
    "excess": presets.boundary_layer_excess,
    "proximal": presets.proximal_excess,
    "proximal_relative": presets.proximal_relative_excess,
}

_ALIASES = {"cfg_baseline": "cfg", "bl": "boundary_layer", "sat": "boundary_layer",
            "boundary_layer_excess": "excess"}

DEFAULT_ARMS = ["cfg", "paper", "excess"]

_FLOAT_FIELDS = ("lam", "k", "phi")

_BOOL_FIELDS = ("store_corrected", "relative_gain", "excess_only")

_BOOLS = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}

def _coerce(field: str, raw: str):
    if field in _FLOAT_FIELDS:
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"{field}={raw!r} is not a number") from None
    if field == "switching":
        if raw not in ("sign", "sat"):
            raise ValueError(f"switching={raw!r} must be 'sign' or 'sat'")
        return raw
    if field == "mode":
        if raw not in ("sliding", "proximal"):
            raise ValueError(f"mode={raw!r} must be 'sliding' or 'proximal'")
        return raw
    if field in _BOOL_FIELDS:
        value = _BOOLS.get(raw.lower())
        if value is None:
            raise ValueError(f"{field}={raw!r} must be true or false")
        return value
    known = ", ".join(_FLOAT_FIELDS + ("switching", "mode") + _BOOL_FIELDS)
    raise ValueError(f"unknown controller field {field!r}; known fields: {known}")

def parse_arm(spec: str, lam: float, k: float) -> Tuple[str, SMCConfig]:
    """Turn one `[name=]preset[:field=value,...]` string into (name, config)."""
    head, _, override_text = spec.partition(":")
    first, eq, second = head.partition("=")
    # "name=preset" splits in two; a bare "preset" leaves everything in `first`.
    name, preset = (first.strip(), second.strip()) if eq else ("", first.strip())
    preset = _ALIASES.get(preset, preset)
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r} in arm {spec!r}; "
                         f"choose from {', '.join(sorted(PRESETS))}")
    name = name or preset
    if "/" in name or "\\" in name:
        raise ValueError(f"arm name {name!r} cannot contain a path separator")

    overrides: Dict[str, object] = {}
    for item in override_text.split(","):
        item = item.strip()
        if not item:
            continue
        field, eq, raw = item.partition("=")
        if not eq:
            raise ValueError(f"override {item!r} in arm {spec!r} must be field=value")
        field = field.strip()
        if field in overrides:
            raise ValueError(f"arm {spec!r} sets {field!r} twice")
        overrides[field] = _coerce(field, raw.strip())

    # Build the preset at the effective lam/k so its derived phi matches, then
    # lay every explicit override on top.
    lam_eff = float(overrides.get("lam", lam))
    k_eff = float(overrides.get("k", k))
    if preset in ("proximal", "proximal_relative"):
        unused = {"lam", "phi", "switching"} & overrides.keys()
        if unused:
            raise ValueError(f"proximal correction does not use {', '.join(sorted(unused))}")
        cfg = PRESETS[preset](k=k_eff)
    else:
        cfg = PRESETS[preset]() if preset == "cfg" else PRESETS[preset](lam_eff, k_eff)
    cfg = replace(cfg, **overrides)
    cfg.validate()
    return name, cfg

def parse_arms(specs, lam: float, k: float) -> Dict[str, SMCConfig]:
    """Resolve every arm spec, rejecting duplicate names (they share a
    directory and would overwrite one another's images)."""
    table: Dict[str, SMCConfig] = {}
    for spec in specs:
        name, cfg = parse_arm(spec, lam, k)
        if name in table:
            raise ValueError(f"two arms are both named {name!r}; "
                             f"give one an explicit name, e.g. myname={spec}")
        table[name] = cfg
    if not table:
        raise ValueError("at least one arm is required")
    return table

def describe_arm(name: str, cfg: SMCConfig) -> str:
    if cfg.is_cfg:
        return f"{name:<12} plain CFG (k = 0, the baseline every other arm is measured against)"
    bits = (["memoryless soft threshold", f"k={cfg.k:g}"] if cfg.mode == "proximal" else
            [f"lam={cfg.lam:g}", f"k={cfg.k:g}", f"switching={cfg.switching}"])
    if cfg.mode == "sliding" and cfg.switching == "sat":
        bits.append(f"phi={cfg.phi:g}")
    if cfg.mode == "sliding" and not cfg.store_corrected:
        bits.append("measured-error memory")
    if cfg.excess_only:
        bits.append("extrapolation only")
    if cfg.relative_gain:
        bits.append("relative gain")
    return f"{name:<12} " + "  ".join(bits)
