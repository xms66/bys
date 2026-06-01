BASE_PRIORS: dict[str, float] = {
    "ice": 0.28,
    "retreat": 0.32,
    "mixed": 0.42,
    "warmup": 0.49,
    "markup": 0.55,
    "climax": 0.48,
}

TREND_ADJUSTMENT: dict[str, float] = {
    "up": 0.03,
    "flat": 0.0,
    "down": -0.04,
}

LIKELIHOODS: dict[str, dict[str, tuple[float, float]]] = {
    "five_day_pattern": {
        "breakout_after_base": (0.72, 0.34),
        "first_pullback": (0.66, 0.40),
        "sideways": (0.50, 0.50),
        "extended_chase": (0.48, 0.56),
        "weak_downtrend": (0.24, 0.70),
    },
    "volume_pattern": {
        "healthy_expansion": (0.66, 0.38),
        "explosive_divergence": (0.46, 0.62),
        "neutral": (0.50, 0.50),
        "shrinking_or_inactive": (0.30, 0.64),
    },
    "message_strength": {
        "strong": (0.74, 0.34),
        "medium": (0.58, 0.45),
        "weak": (0.44, 0.54),
        "none": (0.32, 0.66),
    },
    "concept_strength": {
        "strong": (0.70, 0.36),
        "medium": (0.56, 0.45),
        "weak": (0.38, 0.58),
    },
    "popularity": {
        "top10": (0.68, 0.42),
        "top50": (0.56, 0.48),
        "normal": (0.42, 0.58),
        "cold": (0.30, 0.68),
    },
}

DEFAULT_LIKELIHOOD = (0.50, 0.50)
POSTERIOR_FLOOR = 0.18
POSTERIOR_CEILING = 0.72

