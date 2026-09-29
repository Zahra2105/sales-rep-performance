"""Simulate an inside-sales lead funnel with a known ground truth.

The point of simulating is that the "right answer" is known. A method that
claims to separate rep skill from lead mix can be checked against the skill
values that generated the data, which is impossible with real data.

Story: a subscription-software company routes inbound leads to 18 sales reps.
A lead "converts" when it buys a paid plan. Reps differ in three ways:
  1. the leads they are given (routing is not random),
  2. how they work the lead in the first week (speed, attempts, abandonment),
  3. an unobserved skill (call quality, objection handling, ...).

Only 1 and 2 are visible in the CRM export. 3 is written to a separate
ground-truth file that the analysis never reads.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

SEED = 42
N_LEADS = 60_000
START = pd.Timestamp("2024-01-01")
SNAPSHOT = pd.Timestamp("2025-06-30")

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data"

CHANNELS = ["paid_social", "search", "email", "partner", "referral"]
CHANNEL_P = [0.45, 0.20, 0.12, 0.13, 0.10]
CHANNEL_EFFECT = {"paid_social": 0.0, "search": 0.6, "email": 0.1, "partner": 0.4, "referral": 2.0}

REPS = [f"R{i:02d}" for i in range(1, 19)]


def rep_profiles(rng: np.random.Generator) -> pd.DataFrame:
    """Per-rep routing, behaviour and hidden skill."""
    prof = pd.DataFrame({"rep_id": REPS})
    prof["share_weight"] = rng.uniform(0.6, 1.4, len(REPS))
    prof["metro_share"] = 0.45
    prof["referral_boost"] = 1.0
    prof["median_hours_to_contact"] = rng.uniform(3, 20, len(REPS))
    prof["attempt_rate"] = rng.uniform(2.0, 4.5, len(REPS))
    prof["abandon_rate"] = rng.uniform(0.05, 0.20, len(REPS))
    prof["true_skill"] = rng.normal(0, 0.10, len(REPS))

    def set_rep(rep, **kw):
        for k, v in kw.items():
            prof.loc[prof.rep_id == rep, k] = v

    # The "star": gets the best leads AND works fast AND has some real skill.
    set_rep("R07", metro_share=0.92, referral_boost=2.2, median_hours_to_contact=1.5,
            attempt_rate=4.8, abandon_rate=0.03, true_skill=0.30, share_weight=1.5)
    # Quietly good: ordinary leads, ordinary speed, high hidden skill.
    set_rep("R03", true_skill=0.35, median_hours_to_contact=9, abandon_rate=0.10)
    # Looks bad, mostly because of the leads routed to them.
    set_rep("R12", metro_share=0.12, referral_boost=0.4, true_skill=0.05,
            median_hours_to_contact=6, abandon_rate=0.08)
    # Fast but abandons a lot.
    set_rep("R15", median_hours_to_contact=2, abandon_rate=0.30, true_skill=-0.05)
    return prof


def simulate() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    prof = rep_profiles(rng)

    rep_idx = rng.choice(len(REPS), size=N_LEADS, p=prof.share_weight / prof.share_weight.sum())
    p = prof.iloc[rep_idx].reset_index(drop=True)

    span_days = (SNAPSHOT - START).days
    created_at = START + pd.to_timedelta(rng.uniform(0, span_days, N_LEADS), unit="D")

    region = np.where(rng.uniform(size=N_LEADS) < p.metro_share, "metro", "regional")

    # channel: referral share scaled per rep, then renormalised
    base = np.tile(CHANNEL_P, (N_LEADS, 1))
    base[:, 4] *= p.referral_boost.to_numpy()
    base = base / base.sum(axis=1, keepdims=True)
    u = rng.uniform(size=(N_LEADS, 1))
    channel = np.array(CHANNELS)[(u > base.cumsum(axis=1)).sum(axis=1)]

    lead_score = np.clip(rng.normal(55, 15, N_LEADS) + np.where(channel == "referral", 10, 0), 1, 99).round()

    # first-week behaviour
    abandoned = rng.uniform(size=N_LEADS) < p.abandon_rate
    hours = np.exp(np.log(p.median_hours_to_contact) + rng.normal(0, 1.0, N_LEADS))
    hours_to_first_contact = np.where(abandoned, np.nan, np.round(hours, 1))
    attempts = np.where(abandoned, 0, 1 + rng.poisson(p.attempt_rate - 1))

    # true conversion model (log-odds)
    logit = (
        -4.6
        + np.where(region == "metro", 1.5, 0.0)
        + pd.Series(channel).map(CHANNEL_EFFECT).to_numpy()
        + 0.35 * (lead_score - 55) / 15
        + np.where(hours_to_first_contact <= 1, 0.50, np.where(hours_to_first_contact <= 24, 0.25, 0.0))
        + 0.08 * np.minimum(attempts, 6)
        - 1.2 * abandoned
        + p.true_skill.to_numpy()
    )
    will_convert = rng.uniform(size=N_LEADS) < 1 / (1 + np.exp(-logit))

    # time to conversion: most within a month, a long tail beyond 90 days
    days_to_convert = np.round(rng.gamma(shape=1.1, scale=38, size=N_LEADS) + 1)
    converted_at = created_at + pd.to_timedelta(days_to_convert, unit="D")
    observed = will_convert & (converted_at <= SNAPSHOT)

    leads = pd.DataFrame({
        "lead_id": [f"L{i:06d}" for i in range(N_LEADS)],
        "created_at": created_at.floor("min"),
        "rep_id": p.rep_id,
        "region": region,
        "channel": channel,
        "lead_score": lead_score.astype(int),
        "hours_to_first_contact": hours_to_first_contact,
        "contact_attempts_7d": attempts.astype(int),
        "converted_at": pd.Series(converted_at.floor("min")).where(observed),
    }).sort_values("created_at").reset_index(drop=True)

    truth = prof[["rep_id", "true_skill", "metro_share", "referral_boost",
                  "median_hours_to_contact", "abandon_rate"]]
    return leads, truth


def main() -> None:
    OUT.mkdir(exist_ok=True)
    leads, truth = simulate()
    leads.to_csv(OUT / "leads.csv", index=False)
    truth.to_csv(OUT / "ground_truth_reps.csv", index=False)
    meta = {"seed": SEED, "n_leads": N_LEADS, "snapshot": str(SNAPSHOT.date())}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(leads):,} leads, {leads.converted_at.notna().sum():,} observed conversions")


if __name__ == "__main__":
    main()
