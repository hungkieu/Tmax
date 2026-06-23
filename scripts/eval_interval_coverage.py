"""Honest out-of-sample coverage check for the 80% Tmax prediction interval.

The training/validation pipeline fits the conformal residual quantiles on the
test set and then measures coverage on that *same* test set. By construction the
band between the 10th and 90th residual quantiles contains ~80% of the residuals
it was fit on, so the reported coverage (~80% identical across every cutoff) is
circular and tells us nothing about how the interval generalises to new days.

This script re-uses the saved model bundle to regenerate per-row test
predictions, then estimates coverage *honestly* in two ways:

1. Temporal holdout: fit the residual quantiles on the earlier portion of the
   test dates (calibration) and measure coverage on the later, unseen portion.
   This mimics real deployment: calibrate on the past, forecast the future.
2. K-fold cross-conformal: average coverage across folds where each fold is held
   out from quantile fitting. Lower variance, less sensitive to one split.

For each it reports per-cutoff and overall coverage and mean interval width, and
prints the in-sample (circular) coverage alongside for comparison.

Usage:
    uv run python scripts/eval_interval_coverage.py --config configs/rjtt.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from rksi_tmax.config import load_config
from rksi_tmax.heat_risk import (
    FINAL_TMAX_COLUMN,
    _hhmm_to_minutes,
    _load_model_bundle,
    _predict_remaining_heat,
    load_heat_risk_table,
)

TARGET_COVERAGE = 0.80


def _test_frame_with_predictions(config) -> pd.DataFrame:
    """Rebuild the test split with per-row Tmax predictions, residuals, cutoff."""
    dataset = load_heat_risk_table(config.heat_risk_dataset_parquet).dropna(
        subset=[FINAL_TMAX_COLUMN]
    )
    bundle = _load_model_bundle(config.heat_risk_model_path)
    metrics = bundle["metrics"]
    test = dataset[
        dataset["local_date"].astype(str).between(
            str(metrics["test_start"]), str(metrics["test_end"])
        )
    ].copy()

    remaining = _predict_remaining_heat(bundle, test)
    test["predicted_tmax_c"] = test["tmpc_max_to_cutoff"].to_numpy() + remaining
    test["actual_tmax_c"] = test[FINAL_TMAX_COLUMN].to_numpy()
    test["residual_c"] = test["actual_tmax_c"] - test["predicted_tmax_c"]
    test["cutoff_local"] = test["cutoff_local"].astype(str)
    test["cutoff_minutes"] = test["cutoff_local"].map(_hhmm_to_minutes)
    return test.reset_index(drop=True)


def _coverage_from_quantiles(
    eval_rows: pd.DataFrame,
    quantiles_by_cutoff: dict[str, tuple[float, float]],
    overall_q: tuple[float, float],
) -> pd.DataFrame:
    """Apply fitted residual quantiles to eval rows and flag interval coverage."""
    rows = []
    for _, row in eval_rows.iterrows():
        q_lo, q_hi = quantiles_by_cutoff.get(row["cutoff_local"], overall_q)
        # Practical lower bound is clamped to observed max, matching production.
        low = max(row["tmpc_max_to_cutoff"], row["predicted_tmax_c"] + q_lo)
        high = row["predicted_tmax_c"] + q_hi
        rows.append(
            {
                "cutoff_local": row["cutoff_local"],
                "covered": bool(low <= row["actual_tmax_c"] <= high),
                "width_c": high - low,
            }
        )
    return pd.DataFrame(rows)


def _fit_quantiles(calib: pd.DataFrame) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    overall_q = (
        float(np.quantile(calib["residual_c"], 0.10)),
        float(np.quantile(calib["residual_c"], 0.90)),
    )
    by_cutoff: dict[str, tuple[float, float]] = {}
    for cutoff, group in calib.groupby("cutoff_local"):
        if len(group) < 30:
            continue
        by_cutoff[str(cutoff)] = (
            float(np.quantile(group["residual_c"], 0.10)),
            float(np.quantile(group["residual_c"], 0.90)),
        )
    return by_cutoff, overall_q


def _summarise(flags: pd.DataFrame) -> pd.DataFrame:
    by_cutoff = (
        flags.groupby("cutoff_local")
        .agg(coverage=("covered", "mean"), mean_width_c=("width_c", "mean"), n=("covered", "size"))
        .reset_index()
        .sort_values("cutoff_local")
    )
    overall = pd.DataFrame(
        [{
            "cutoff_local": "OVERALL",
            "coverage": flags["covered"].mean(),
            "mean_width_c": flags["width_c"].mean(),
            "n": len(flags),
        }]
    )
    return pd.concat([by_cutoff, overall], ignore_index=True)


def _print_table(title: str, summary: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(f"{'cutoff':>8} {'coverage':>9} {'target':>7} {'width_c':>8} {'n':>5}")
    for _, r in summary.iterrows():
        gap = r["coverage"] - TARGET_COVERAGE
        flag = "OK" if abs(gap) <= 0.03 else ("LOW" if gap < 0 else "WIDE")
        print(
            f"{r['cutoff_local']:>8} {r['coverage']*100:>8.1f}% "
            f"{TARGET_COVERAGE*100:>6.0f}% {r['mean_width_c']:>8.2f} {int(r['n']):>5}  {flag}"
        )


def temporal_holdout(test: pd.DataFrame, calib_frac: float) -> pd.DataFrame:
    dates = np.array(sorted(test["local_date"].astype(str).unique()))
    split_at = dates[int(len(dates) * calib_frac)]
    calib = test[test["local_date"].astype(str) < split_at]
    evalset = test[test["local_date"].astype(str) >= split_at]
    by_cutoff, overall_q = _fit_quantiles(calib)
    flags = _coverage_from_quantiles(evalset, by_cutoff, overall_q)
    print(
        f"  calibration dates < {split_at} (n={len(calib)}), "
        f"eval dates >= {split_at} (n={len(evalset)})"
    )
    return _summarise(flags)


def kfold_cross_conformal(test: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    dates = np.array(sorted(test["local_date"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(dates)
    folds = np.array_split(dates, k)
    all_flags = []
    for fold in folds:
        eval_mask = test["local_date"].astype(str).isin(set(fold))
        calib = test[~eval_mask]
        evalset = test[eval_mask]
        by_cutoff, overall_q = _fit_quantiles(calib)
        all_flags.append(_coverage_from_quantiles(evalset, by_cutoff, overall_q))
    return _summarise(pd.concat(all_flags, ignore_index=True))


def in_sample(test: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the pipeline's circular coverage: fit and eval on same rows."""
    by_cutoff, overall_q = _fit_quantiles(test)
    flags = _coverage_from_quantiles(test, by_cutoff, overall_q)
    return _summarise(flags)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rjtt.yaml")
    parser.add_argument("--calib-frac", type=float, default=0.6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    test = _test_frame_with_predictions(config)
    print(f"Station: {config.station}  test rows: {len(test)}  "
          f"dates: {test['local_date'].min()} .. {test['local_date'].max()}")
    print(f"Point Tmax MAE on test: {test['residual_c'].abs().mean():.3f} C")

    _print_table("[1] IN-SAMPLE (circular -- what the report currently shows)", in_sample(test))
    print("\n  >>> identical ~80% per cutoff is the tell: fit and eval on the same rows.")

    print("\n[2] TEMPORAL HOLDOUT (honest: calibrate on past, evaluate on future)")
    _print_table("    out-of-sample coverage", temporal_holdout(test, args.calib_frac))

    _print_table(
        f"[3] {args.folds}-FOLD CROSS-CONFORMAL (honest, lower variance)",
        kfold_cross_conformal(test, args.folds, args.seed),
    )

    print(
        "\nReading: coverage near 80% (OK) = interval well calibrated. "
        "LOW = interval too narrow (overconfident); WIDE = too wide (underconfident)."
    )


if __name__ == "__main__":
    main()
