"""Is the top sales rep better, or just better supplied?

Reads data/leads.csv (the CRM export) and writes:
    reports/results.json      every number quoted in the report
    reports/figures/*.png     charts

data/ground_truth_reps.csv is read ONLY in the final validation step, to
check the method against the skill values that generated the data.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"

SNAPSHOT = pd.Timestamp("2025-06-30")
WINDOW_DAYS = 90
FOCUS_REP = "R07"
RNG = np.random.default_rng(7)

NAVY, RED, GREY, LIGHT = "#1F374C", "#7B2D2B", "#8A949C", "#D9DEE3"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#444", "axes.titleweight": "bold",
    "axes.titlesize": 11, "figure.dpi": 150,
})

FORMULA_MIX = "converted ~ C(rep_id) + C(region) + C(channel) + score_std"
FORMULA_FULL = FORMULA_MIX + " + C(speed, Treatment('over_24h')) + attempts_capped"  # "never" = abandoned
FORMULA_NO_REP = "converted ~ C(region) + C(channel) + score_std + C(speed) + attempts_capped"


# ---------------------------------------------------------------- data prep
def load() -> pd.DataFrame:
    d = pd.read_csv(DATA / "leads.csv", parse_dates=["created_at", "converted_at"])
    d["days_to_convert"] = (d.converted_at - d.created_at).dt.days
    d["abandoned"] = d.hours_to_first_contact.isna().astype(int)
    d["speed"] = pd.cut(d.hours_to_first_contact, [-0.1, 1, 24, np.inf],
                        labels=["within_1h", "1_to_24h", "over_24h"]).astype(str)
    d.loc[d.abandoned == 1, "speed"] = "never"
    d["attempts_capped"] = d.contact_attempts_7d.clip(upper=6)
    d["score_std"] = (d.lead_score - 55) / 15
    d["is_mature"] = d.created_at <= SNAPSHOT - pd.Timedelta(days=WINDOW_DAYS)
    d["converted"] = (d.days_to_convert <= WINDOW_DAYS).astype(int)
    return d


# ---------------------------------------------------------------- steps
def window_choice(d: pd.DataFrame) -> dict:
    # only leads old enough that a 180-day conversion could have been seen
    old = d[d.created_at <= SNAPSHOT - pd.Timedelta(days=180)]
    days = old.days_to_convert.dropna()
    res = {
        "n_conversions": int(len(days)),
        "median": float(days.median()),
        "p75": float(days.quantile(0.75)),
        "p80": float(days.quantile(0.80)),
        "p95": float(days.quantile(0.95)),
        "share_within_window": float((days <= WINDOW_DAYS).mean()),
    }
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(days.clip(upper=200), bins=40, color=LIGHT, edgecolor="white")
    ax.hist(days[days <= WINDOW_DAYS], bins=np.linspace(0, 200, 41), color=NAVY, edgecolor="white")
    ax.axvline(WINDOW_DAYS, color=RED, lw=2)
    ax.text(WINDOW_DAYS + 3, ax.get_ylim()[1] * 0.85,
            f"{WINDOW_DAYS}-day window\ncovers {res['share_within_window']:.0%}", color=RED)
    ax.set_title("Days from lead to purchase (leads with 180+ days of history)")
    ax.set_xlabel("days"); ax.set_ylabel("conversions")
    fig.tight_layout(); fig.savefig(FIG / "01_window.png"); plt.close(fig)
    return res


def raw_ranking(m: pd.DataFrame) -> pd.DataFrame:
    r = m.groupby("rep_id").agg(leads=("converted", "size"), conversions=("converted", "sum"),
                                rate=("converted", "mean"),
                                metro_share=("region", lambda s: (s == "metro").mean()),
                                referral_share=("channel", lambda s: (s == "referral").mean()))
    return r.sort_values("rate", ascending=False)


def sensitivity(m: pd.DataFrame) -> list[dict]:
    cuts = [
        ("All mature leads", m),
        ("Metro only", m[m.region == "metro"]),
        ("Metro, excluding referrals", m[(m.region == "metro") & (m.channel != "referral")]),
        ("Metro, paid social only", m[(m.region == "metro") & (m.channel == "paid_social")]),
    ]
    out = []
    for label, s in cuts:
        f, o = s[s.rep_id == FOCUS_REP], s[s.rep_id != FOCUS_REP]
        out.append({"cut": label, "focus_leads": int(len(f)), "focus_rate": float(f.converted.mean()),
                    "others_leads": int(len(o)), "others_rate": float(o.converted.mean()),
                    "ratio": float(f.converted.mean() / o.converted.mean())})
    return out


def fit(formula: str, df: pd.DataFrame):
    return smf.logit(formula, data=df).fit(disp=False, maxiter=200)


def standardized_rates(model, m: pd.DataFrame) -> pd.Series:
    """Rate each rep would have on the SAME leads (all mature leads)."""
    rates = {}
    for rep in sorted(m.rep_id.unique()):
        cf = m.assign(rep_id=rep)
        rates[rep] = float(model.predict(cf).mean())
    return pd.Series(rates)


def temporal_validation(m: pd.DataFrame) -> dict:
    cutoff = m.created_at.quantile(0.75)
    train, test = m[m.created_at <= cutoff], m[m.created_at > cutoff]
    out = {"cutoff": str(cutoff.date()), "train_n": int(len(train)), "test_n": int(len(test))}
    for name, f in [("lead_mix_only", FORMULA_MIX), ("mix_plus_behaviour", FORMULA_FULL)]:
        p = fit(f, train).predict(test)
        out[name] = {"roc_auc": float(roc_auc_score(test.converted, p)),
                     "pr_auc": float(average_precision_score(test.converted, p))}
    # calibration of the full model by decile
    p = fit(FORMULA_FULL, train).predict(test)
    cal = pd.DataFrame({"p": p, "y": test.converted}).assign(decile=lambda x: pd.qcut(x.p, 10, labels=False))
    cal = cal.groupby("decile").agg(predicted=("p", "mean"), observed=("y", "mean"))
    out["calibration"] = cal.round(4).reset_index().to_dict(orient="records")
    return out


def decompose(m: pd.DataFrame, full) -> dict:
    """Walk from the others' rate to the focus rep's rate in three steps."""
    focus, others = m[m.rep_id == FOCUS_REP], m[m.rep_id != FOCUS_REP]
    other_reps = sorted(others.rep_id.unique())
    beh_cols = ["speed", "attempts_capped", "abandoned"]

    def avg_rep_pred(df):
        return np.mean([full.predict(df.assign(rep_id=r)).mean() for r in other_reps])

    # focus leads, but with behaviour drawn from other reps' leads
    donor = others[beh_cols].sample(len(focus), replace=True, random_state=11).reset_index(drop=True)
    focus_other_beh = focus.reset_index(drop=True).copy()
    focus_other_beh[beh_cols] = donor

    base = float(others.converted.mean())
    s1 = float(avg_rep_pred(focus_other_beh))          # focus leads, typical behaviour, typical rep
    s2 = float(avg_rep_pred(focus))                    # + focus rep's behaviour
    actual = float(focus.converted.mean())
    return {"others_rate": base, "lead_mix": s1 - base, "behaviour": s2 - s1,
            "unexplained": actual - s2, "focus_rate": actual}


def scenarios(m: pd.DataFrame, full) -> dict:
    """Projected conversions per 1,000 leads under operating changes."""
    base = full.predict(m).mean()
    fast = m.copy()
    fast.loc[fast.abandoned == 0, "speed"] = "within_1h"
    no_abandon = m.copy()
    ab = no_abandon.abandoned == 1
    no_abandon.loc[ab, "abandoned"] = 0
    no_abandon.loc[ab, "speed"] = "1_to_24h"
    no_abandon.loc[ab, "attempts_capped"] = 3
    both = no_abandon.copy()
    both["speed"] = "within_1h"
    res = {"baseline": float(base)}
    for k, df in [("contact_within_1h", fast), ("no_abandoned_leads", no_abandon), ("both", both)]:
        res[k] = float(full.predict(df).mean())
    return res


def pilot_size(p0: float, p1: float) -> int:
    es = proportion_effectsize(p1, p0)
    return int(np.ceil(NormalIndPower().solve_power(effect_size=es, alpha=0.05, power=0.8, ratio=1)))


# ---------------------------------------------------------------- charts
def chart_ranking(raw: pd.DataFrame, std: pd.Series) -> None:
    order = raw.index.tolist()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(order))
    ax.barh(y + 0.2, raw.rate[order], height=0.4, color=LIGHT, label="raw rate (own leads)")
    ax.barh(y - 0.2, std[order], height=0.4, color=NAVY, label="adjusted rate (same leads for everyone)")
    i = order.index(FOCUS_REP)
    ax.barh(i - 0.2, std[FOCUS_REP], height=0.4, color=RED)
    ax.set_yticks(y, order); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_title("90-day conversion by rep: raw vs adjusted for lead mix")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "02_ranking.png"); plt.close(fig)


def chart_sensitivity(sens: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    labels = [s["cut"] for s in sens]; ratios = [s["ratio"] for s in sens]
    ax.barh(labels, ratios, color=[NAVY] * (len(sens) - 1) + [RED])
    ax.axvline(1, color="#444", lw=1)
    for i, r in enumerate(ratios):
        ax.text(r + 0.05, i, f"{r:.2f}×", va="center")
    ax.invert_yaxis(); ax.set_xlim(0, max(ratios) * 1.2)
    ax.set_title(f"{FOCUS_REP}'s conversion rate ÷ everyone else's, as leads get more alike")
    fig.tight_layout(); fig.savefig(FIG / "03_sensitivity.png"); plt.close(fig)


def chart_waterfall(dec: dict) -> None:
    steps = [("Other reps", dec["others_rate"], NAVY, True),
             ("Better\nleads", dec["lead_mix"], GREY, False),
             ("Faster, fuller\nfollow-up", dec["behaviour"], NAVY, False),
             ("Not explained\nby the data", dec["unexplained"], RED, False),
             (FOCUS_REP, dec["focus_rate"], NAVY, True)]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    running = 0.0
    for i, (lab, v, c, total) in enumerate(steps):
        bottom = 0 if total else running
        ax.bar(i, v, bottom=bottom, color=c, width=0.6)
        top = v if total else running + v
        ax.text(i, max(bottom, top) + 0.004, f"{v*100:+.1f} pts" if not total else f"{v:.1%}",
                ha="center", fontsize=9)
        running = v if total else running + v
    ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=8.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_title(f"Where {FOCUS_REP}'s advantage comes from")
    fig.tight_layout(); fig.savefig(FIG / "04_decomposition.png"); plt.close(fig)


def chart_truth(truth: pd.DataFrame, raw: pd.DataFrame, skill_est: pd.Series) -> None:
    t = truth.set_index("rep_id")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
    for ax, x, title in [(axes[0], raw.rate, "Raw conversion rate"),
                         (axes[1], skill_est, "Adjusted rate (same leads and behaviour)")]:
        x = x.reindex(t.index)
        ax.scatter(x, t.true_skill, color=NAVY, s=28)
        for rep in ["R07", "R03", "R12"]:
            ax.scatter(x[rep], t.true_skill[rep], color=RED, s=40)
            ax.annotate(rep, (x[rep], t.true_skill[rep]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        rho = spearmanr(x, t.true_skill).statistic
        ax.set_title(f"{title}\nrank correlation with true skill: {rho:.2f}", fontsize=9.5)
    axes[0].set_ylabel("true (hidden) skill")
    for a in axes: a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    fig.tight_layout(); fig.savefig(FIG / "06_ground_truth.png"); plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    d = load()
    m = d[d.is_mature].copy()

    res: dict = {"n_leads": int(len(d)), "n_mature": int(len(m)),
                 "n_conversions_90d": int(m.converted.sum()), "overall_rate": float(m.converted.mean()),
                 "n_reps": int(m.rep_id.nunique()), "focus_rep": FOCUS_REP}
    res["window"] = window_choice(d)

    raw = raw_ranking(m)
    res["raw_ranking"] = raw.round(4).reset_index().to_dict(orient="records")
    f, o = m[m.rep_id == FOCUS_REP], m[m.rep_id != FOCUS_REP]
    res["focus_raw"] = {"rate": float(f.converted.mean()), "others_rate": float(o.converted.mean()),
                        "ratio": float(f.converted.mean() / o.converted.mean()),
                        "metro_share": float((f.region == "metro").mean()),
                        "others_metro_share": float((o.region == "metro").mean()),
                        "referral_share": float((f.channel == "referral").mean()),
                        "others_referral_share": float((o.channel == "referral").mean())}
    res["segment_rates"] = {
        "metro": float(m[m.region == "metro"].converted.mean()),
        "regional": float(m[m.region == "regional"].converted.mean()),
        "referral": float(m[m.channel == "referral"].converted.mean()),
        "paid_social": float(m[m.channel == "paid_social"].converted.mean()),
    }

    res["sensitivity"] = sensitivity(m)
    chart_sensitivity(res["sensitivity"])

    mix = fit(FORMULA_MIX, m)
    full = fit(FORMULA_FULL, m)
    std_mix = standardized_rates(mix, m)
    std_full = standardized_rates(full, m)
    res["adjusted_rates_mix"] = std_mix.round(4).to_dict()
    res["adjusted_rates_full"] = std_full.round(4).to_dict()
    chart_ranking(raw, std_mix)

    res["behaviour_effects"] = {k: float(v) for k, v in full.params.items()
                                if k.startswith(("C(speed", "attempts"))}
    res["validation"] = temporal_validation(m)

    dec = decompose(m, full)
    res["decomposition"] = dec
    chart_waterfall(dec)

    sc = scenarios(m, full)
    res["scenarios"] = sc
    res["pilot_n_per_arm"] = pilot_size(sc["baseline"], sc["both"])
    res["speed_rates"] = m.groupby("speed").converted.agg(["size", "mean"]).round(4).reset_index().to_dict(orient="records")

    # ---- validation against the hidden truth (only place it is read)
    truth = pd.read_csv(DATA / "ground_truth_reps.csv")
    skill_est = std_full  # rep effect net of mix and behaviour
    t = truth.set_index("rep_id").true_skill
    res["ground_truth"] = {
        "spearman_raw_vs_truth": float(spearmanr(raw.rate.reindex(t.index), t).statistic),
        "spearman_adjusted_vs_truth": float(spearmanr(skill_est.reindex(t.index), t).statistic),
        "top_by_raw": raw.index[0],
        "top_by_adjusted": skill_est.idxmax(),
        "true_top": t.idxmax(),
        "rank_of_R12_raw": int(list(raw.index).index("R12") + 1),
        "rank_of_R12_adjusted": int(list(skill_est.sort_values(ascending=False).index).index("R12") + 1),
    }
    chart_truth(truth, raw, skill_est)

    (REPORTS / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ["overall_rate", "focus_raw", "decomposition", "scenarios",
                                          "pilot_n_per_arm", "ground_truth"]}, indent=2, default=str))
    print(json.dumps(res["validation"], indent=1)[:600])
    print(json.dumps(res["sensitivity"], indent=1))
    print(json.dumps(res["window"], indent=1))


if __name__ == "__main__":
    main()
