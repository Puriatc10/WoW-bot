import math
import random
from dataclasses import dataclass
from statistics import NormalDist


class IntervalError(Exception):
    """Raised when interval sampling or operation fails (e.g. rejection limit reached)."""


@dataclass(frozen=True)
class IntervalConfig:
    mu: float = -2.0
    sigma: float = 0.6
    clip_low: float = 0.02
    clip_high: float = 1.5
    max_rejection_attempts: int = 1000

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu):
            raise ValueError(f"mu must be finite, got {self.mu}")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0.0, got {self.sigma}")
        if self.clip_low <= 0.0:
            raise ValueError(f"clip_low must be > 0.0, got {self.clip_low}")
        if self.clip_high <= self.clip_low:
            raise ValueError(
                f"clip_high must be > clip_low ({self.clip_low}), got {self.clip_high}"
            )
        if self.max_rejection_attempts < 1:
            raise ValueError(
                f"max_rejection_attempts must be >= 1, got {self.max_rejection_attempts}"
            )


def sample_interval(
    rng: random.Random,
    config: IntervalConfig | None = None,
) -> float:
    """Draw a sample from the truncated lognormal distribution [clip_low, clip_high].

    Draws lognormal samples via exp(rng.gauss(mu, sigma)) and rejects values outside
    [clip_low, clip_high] up to config.max_rejection_attempts times.
    """
    cfg = config if config is not None else IntervalConfig()

    for _ in range(cfg.max_rejection_attempts):
        draw = math.exp(rng.gauss(cfg.mu, cfg.sigma))
        if cfg.clip_low <= draw <= cfg.clip_high:
            return draw

    raise IntervalError(
        f"Exhausted max_rejection_attempts ({cfg.max_rejection_attempts}) without "
        f"drawing a sample in range [{cfg.clip_low}, {cfg.clip_high}] for "
        f"mu={cfg.mu}, sigma={cfg.sigma}."
    )


def theoretical_cv(config: IntervalConfig) -> float:
    """Returns coefficient of variation sqrt(exp(sigma^2) - 1) for the untruncated lognormal.

    Note: Truncation can only reduce CV. This serves as an upper bound approximation.
    """
    return math.sqrt(math.exp(config.sigma**2) - 1.0)


def theoretical_mean(config: IntervalConfig) -> float:
    """Returns mean exp(mu + sigma^2 / 2) for the untruncated lognormal."""
    return math.exp(config.mu + (config.sigma**2) / 2.0)


def theoretical_median(config: IntervalConfig) -> float:
    """Returns median exp(mu) for the untruncated lognormal."""
    return math.exp(config.mu)


def is_in_support(x: float, config: IntervalConfig | None = None) -> bool:
    """Returns True iff clip_low <= x <= clip_high."""
    cfg = config if config is not None else IntervalConfig()
    return cfg.clip_low <= x <= cfg.clip_high


def cdf(x: float, config: IntervalConfig | None = None) -> float:
    """Returns the CDF of the TRUNCATED lognormal at x."""
    cfg = config if config is not None else IntervalConfig()

    if x <= cfg.clip_low:
        return 0.0
    if x >= cfg.clip_high:
        return 1.0

    normal_dist = NormalDist()
    f_x = normal_dist.cdf((math.log(x) - cfg.mu) / cfg.sigma)
    f_lo = normal_dist.cdf((math.log(cfg.clip_low) - cfg.mu) / cfg.sigma)
    f_hi = normal_dist.cdf((math.log(cfg.clip_high) - cfg.mu) / cfg.sigma)

    if f_hi <= f_lo:
        # Extreme numerical truncation boundary safeguard
        return 0.0

    return (f_x - f_lo) / (f_hi - f_lo)


def logpdf(x: float, config: IntervalConfig | None = None) -> float:
    """Returns the log-density of the TRUNCATED lognormal at x."""
    cfg = config if config is not None else IntervalConfig()

    if not is_in_support(x, cfg):
        return float("-inf")

    normal_dist = NormalDist()
    f_lo = normal_dist.cdf((math.log(cfg.clip_low) - cfg.mu) / cfg.sigma)
    f_hi = normal_dist.cdf((math.log(cfg.clip_high) - cfg.mu) / cfg.sigma)
    denom = f_hi - f_lo

    if denom <= 0.0:
        return float("-inf")

    log_x = math.log(x)
    logpdf_lognormal = (
        -log_x
        - math.log(cfg.sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - ((log_x - cfg.mu) ** 2) / (2.0 * (cfg.sigma**2))
    )

    return logpdf_lognormal - math.log(denom)
