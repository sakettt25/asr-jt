"""
Q1 live report generator.
Builds artifacts/q1/report.json from existing live outputs instead of hardcoded values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ARTIFACTS_DIR = Path("./artifacts/q1")
ERROR_CSV_PATH = ARTIFACTS_DIR / "error_samples.csv"
REPORT_PATH = ARTIFACTS_DIR / "report.json"


CATEGORY_DESCRIPTIONS = {
    "phoneme_confusion": "Acoustically similar phonemes are confused in decoding.",
    "word_boundary": "Word segmentation errors such as fusion or split tokens.",
    "english_loanword": "Loanword rendering inconsistencies in Hindi context.",
    "accent_dialect": "Regional pronunciation variation affects recognition.",
    "number_handling": "Number words/digits are inconsistently recognized.",
}


FIX_TEMPLATES = {
    "phoneme_confusion": {
        "fix_name": "Phoneme-aware post-correction",
        "approach": "Use confusion-pair dictionary and context scoring for common Devanagari phoneme substitutions.",
        "effort": "Medium",
    },
    "word_boundary": {
        "fix_name": "Boundary restoration pass",
        "approach": "Apply token-merge/split rules based on Hindi lexicon and confidence-guided boundaries.",
        "effort": "Medium",
    },
    "english_loanword": {
        "fix_name": "Roman→Devanagari loanword correction",
        "approach": "Detect Roman tokens and transliterate/tag loanwords to match submission conventions.",
        "effort": "Low",
    },
    "accent_dialect": {
        "fix_name": "Dialect-aware adaptation",
        "approach": "Add accent-diverse samples and targeted fine-tuning for high-variance phones.",
        "effort": "High",
    },
    "number_handling": {
        "fix_name": "Numeric normalization layer",
        "approach": "Normalize number variants (word/digit) before and after decoding for consistency.",
        "effort": "Low",
    },
}


def _safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _compute_taxonomy(error_df: pd.DataFrame) -> Dict[str, Any]:
    if error_df.empty or "category" not in error_df.columns:
        return {"categories": [], "total_error_categories": 0}

    total = len(error_df)
    categories: List[Dict[str, Any]] = []

    for category, group in error_df.groupby("category", dropna=False):
        category_key = str(category)
        freq = int(len(group))
        samples: List[Dict[str, Any]] = []
        for _, row in group.head(5).iterrows():
            samples.append(
                {
                    "reference": str(row.get("reference", "")),
                    "output": str(row.get("hypothesis", row.get("predicted", ""))),
                    "cause": str(row.get("root_cause", "")),
                }
            )

        categories.append(
            {
                "id": category_key,
                "name": category_key.replace("_", " ").title(),
                "frequency": freq,
                "frequency_pct": round((freq / total) * 100.0, 2),
                "description": CATEGORY_DESCRIPTIONS.get(
                    category_key,
                    "Error category discovered from live sampled hypotheses.",
                ),
                "examples": samples,
            }
        )

    categories.sort(key=lambda item: item["frequency"], reverse=True)

    return {
        "categories": categories,
        "total_error_categories": len(categories),
    }


def _compute_top_fixes(categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fixes: List[Dict[str, Any]] = []
    for rank, cat in enumerate(categories[:3], start=1):
        cat_id = cat["id"]
        template = FIX_TEMPLATES.get(
            cat_id,
            {
                "fix_name": f"Mitigate {cat_id}",
                "approach": "Targeted post-processing and additional supervised samples.",
                "effort": "Medium",
            },
        )
        fixes.append(
            {
                "rank": rank,
                "fix_name": template["fix_name"],
                "target_errors": cat_id,
                "approach": template["approach"],
                "potential_impact_pct": round(float(cat["frequency_pct"]), 2),
                "effort": template["effort"],
            }
        )
    return fixes


def generate_q1_live_report() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ERROR_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing live error samples CSV: {ERROR_CSV_PATH}")

    error_df = pd.read_csv(ERROR_CSV_PATH)

    existing_report = _safe_load_json(REPORT_PATH)
    results = existing_report.get("results", [])
    dataset = existing_report.get("dataset", "Live Hindi ASR evaluation")
    metric = existing_report.get("metric", "Word Error Rate (WER %)")

    taxonomy = _compute_taxonomy(error_df)
    top_fixes = _compute_top_fixes(taxonomy["categories"])

    severity_counts: Dict[str, int] = {}
    if "severity" in error_df.columns:
        severity_counts = {
            str(level): int(count)
            for level, count in error_df["severity"].value_counts().to_dict().items()
        }

    implemented_fix_source = existing_report.get("implemented_fix", {})
    implemented_fix = {
        "name": implemented_fix_source.get("name", "Not yet evaluated"),
        "target_subset_size": int(implemented_fix_source.get("target_subset_size", 0) or 0),
        "wer_before": implemented_fix_source.get("results", {}).get("wer_before"),
        "wer_after": implemented_fix_source.get("results", {}).get("wer_after"),
        "delta_pp": implemented_fix_source.get("results", {}).get("delta_pp"),
        "note": "Derived from current live artifacts; rerun Q1 training/eval for refreshed values.",
    }

    preprocessing = existing_report.get(
        "preprocessing_pipeline",
        {
            "steps": [
                "Used available live ASR artifacts and sampled error CSV for taxonomy and fix planning.",
                "No synthetic/mock examples injected during report generation.",
            ]
        },
    )

    report = {
        "dataset": dataset,
        "metric": metric,
        "results": results,
        "preprocessing": preprocessing,
        "preprocessing_pipeline": preprocessing,
        "sampling": {
            "total_error_samples": int(len(error_df)),
            "strategy": "Live sampled error rows from artifacts/q1/error_samples.csv",
            "severity_distribution": severity_counts,
            "error_samples_csv": str(ERROR_CSV_PATH).replace("\\", "/"),
        },
        "error_taxonomy": taxonomy,
        "top_fixes": top_fixes,
        "proposed_fixes": top_fixes,
        "implementation_results": implemented_fix,
        "implemented_fix": implemented_fix,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    return REPORT_PATH


if __name__ == "__main__":
    out_path = generate_q1_live_report()
    print(f"✓ Q1 live report generated: {out_path}")
