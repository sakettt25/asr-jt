"""
Question 4: Lattice-Based WER Evaluation
Josh Talks | AI Researcher Intern - Speech & Audio Assignment

Theory:
  Standard WER compares model output against a single rigid reference string.
  This unfairly penalises models when the reference itself is wrong, when
  multiple valid spellings exist, or when number words and digits are
  interchangeable.

  A Lattice addresses this by constructing a sequential list of "bins",
  where each bin contains all VALID lexical alternatives at a given alignment
  position. WER is then computed as the minimum edit distance between the
  model output and the BEST path through the lattice.

Alignment Unit Justification:
  We use WORD-level alignment because:
    (a) Hindi ASR evaluation is standardised at word level.
    (b) Sub-word units create spurious matches across morphological boundaries.
    (c) Phrase-level is too coarse and misses per-word substitution errors.
    (d) Word-level aligns with how human annotators assess correctness.
"""

import re, json, unicodedata
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class LatticeBin:
    """
    One position in the lattice — corresponds to one aligned token slot.
    Contains all valid surface forms for that position.
    """
    position:   int
    variants:   Set[str] = field(default_factory=set)
    is_deleted: bool = False     # True if this position was deleted in some outputs

    def add(self, word: str):
        if word:
            # Normalise: lowercase + NFC Unicode + strip whitespace
            normalised = unicodedata.normalize("NFC", word.strip().lower())
            self.variants.add(normalised)

    def matches(self, word: str) -> bool:
        normalised = unicodedata.normalize("NFC", word.strip().lower())
        return normalised in self.variants


Lattice = List[LatticeBin]


# ─────────────────────────────────────────────────────────────
# VARIANT EXPANSION RULES
# ─────────────────────────────────────────────────────────────

# Number word ↔ digit equivalences for Hindi
HINDI_NUMBER_VARIANTS: Dict[str, str] = {
    "एक": "1",    "दो": "2",    "तीन": "3",    "चार": "4",
    "पाँच": "5",  "पांच": "5",  "छः": "6",     "छह": "6",
    "सात": "7",   "आठ": "8",    "नौ": "9",     "दस": "10",
    "ग्यारह": "11", "बारह": "12", "तेरह": "13",  "चौदह": "14",
    "पंद्रह": "15", "बीस": "20",  "पच्चीस": "25", "तीस": "30",
    "पचास": "50", "सौ": "100",  "हज़ार": "1000", "हजार": "1000",
    "लाख": "100000",
}

# Devanagari digit → ASCII equivalences
DEVA_DIGIT_VARIANTS: Dict[str, str] = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

# Common spelling variants in Hindi (anusvara/chandrabindu interchangeability,
# nukta optional, etc.)
def get_spelling_variants(word: str) -> Set[str]:
    """
    Generate valid orthographic variants of a Hindi word.
    Handles:
      - anusvara/chandrabindu interchangeability (ं/ँ)
      - optional nukta (़)
      - Devanagari vs ASCII digits
    """
    variants = {word}
    nfc = unicodedata.normalize("NFC", word)
    variants.add(nfc)
    nfd = unicodedata.normalize("NFD", word)
    variants.add(nfd)

    # Anusvara ↔ chandrabindu
    variants.add(word.replace("ं", "ँ"))
    variants.add(word.replace("ँ", "ं"))

    # Optional nukta removal (फ़ → फ, ज़ → ज)
    variants.add(word.replace("़", ""))

    # Devanagari digits → ASCII
    ascii_form = word
    for deva, asc in DEVA_DIGIT_VARIANTS.items():
        ascii_form = ascii_form.replace(deva, asc)
    variants.add(ascii_form)

    # Number word ↔ digit
    lower = word.lower()
    if lower in HINDI_NUMBER_VARIANTS:
        variants.add(HINDI_NUMBER_VARIANTS[lower])

    # Reverse: digit → number word
    for word_form, digit in HINDI_NUMBER_VARIANTS.items():
        if word == digit:
            variants.add(word_form)

    # Common synonym pairs (lexical alternatives)
    SYNONYMS: Dict[str, Set[str]] = {
        "किताबें": {"पुस्तकें", "किताबें"},
        "पुस्तकें": {"किताबें", "पुस्तकें"},
        "खरीदीं": {"खरीदी", "खरीदीं"},
        "खरीदी":  {"खरीदीं", "खरीदी"},
        "उसने":   {"उसने", "उन्होंने"},
        "बहुत":   {"बहुत", "काफ़ी", "काफी"},
        "काफी":   {"काफ़ी", "काफी", "बहुत"},
        "काफ़ी":  {"काफ़ी", "काफी", "बहुत"},
        "ठीक":    {"ठीक", "सही"},
        "सही":    {"सही", "ठीक"},
        "देखना":  {"देखना", "देखने"},
        "आना":    {"आना", "आने"},
    }
    if word in SYNONYMS:
        variants.update(SYNONYMS[word])

    return {v for v in variants if v}


# ─────────────────────────────────────────────────────────────
# LATTICE CONSTRUCTION
# ─────────────────────────────────────────────────────────────

class LatticeBuilder:
    """
    Constructs a lattice from:
      - A human reference transcription
      - Multiple ASR model outputs

    Algorithm:
      1.  Tokenise all strings into word lists.
      2.  Align all model outputs against the reference using
          dynamic-programming Needleman-Wunsch (word-level).
      3.  For each aligned position:
            a. Collect all surface forms from all models and the reference.
            b. Expand each form with spelling variants and synonym alternatives.
            c. Detect reference errors: if ≥ TRUST_THRESHOLD fraction of
               models agree on a form that differs from the reference,
               add the model-agreed form to the bin (trust model over reference).
      4.  Output a Lattice: list of LatticeBin objects.

    Trust policy (when to override reference):
      If ≥ 60% of valid model outputs agree on a word form that differs
      from the reference at that position, we add it to the bin as a valid
      alternative. This handles cases where the human reference contains
      a transcription error.
    """

    TRUST_THRESHOLD = 0.60      # fraction of models that must agree to override ref
    GAP_CHAR        = "<GAP>"   # insertion/deletion placeholder

    def __init__(self):
        pass

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenise on whitespace after NFC normalisation."""
        text = unicodedata.normalize("NFC", text.strip())
        return [t for t in text.split() if t]

    @staticmethod
    def _edit_align(ref_tokens: List[str],
                    hyp_tokens: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Global Needleman-Wunsch alignment (word-level).
        Returns list of (ref_word, hyp_word) pairs where None = gap.
        Match/mismatch cost: 0/-1, gap cost: -1.
        """
        R, H = len(ref_tokens), len(hyp_tokens)
        # DP table
        dp = np.zeros((R + 1, H + 1), dtype=int)
        for i in range(R + 1):
            dp[i][0] = -i
        for j in range(H + 1):
            dp[0][j] = -j

        for i in range(1, R + 1):
            for j in range(1, H + 1):
                match = 0 if (unicodedata.normalize("NFC", ref_tokens[i-1].lower()) ==
                               unicodedata.normalize("NFC", hyp_tokens[j-1].lower())) else -1
                dp[i][j] = max(
                    dp[i-1][j-1] + match,
                    dp[i-1][j]   - 1,
                    dp[i][j-1]   - 1,
                )

        # Traceback
        alignment = []
        i, j = R, H
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                match = 0 if (unicodedata.normalize("NFC", ref_tokens[i-1].lower()) ==
                               unicodedata.normalize("NFC", hyp_tokens[j-1].lower())) else -1
                if dp[i][j] == dp[i-1][j-1] + match:
                    alignment.append((ref_tokens[i-1], hyp_tokens[j-1]))
                    i -= 1; j -= 1
                    continue
            if i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] - 1):
                alignment.append((ref_tokens[i-1], None))
                i -= 1
            else:
                alignment.append((None, hyp_tokens[j-1]))
                j -= 1

        alignment.reverse()
        return alignment

    def build(self, reference: str, model_outputs: Dict[str, str]) -> Lattice:
        """
        Build a lattice from one reference and N model outputs.

        Args:
            reference:     Human reference transcription string.
            model_outputs: Dict mapping model_name → hypothesis string.

        Returns:
            Lattice (list of LatticeBin).
        """
        ref_tokens     = self._tokenize(reference)
        model_token_lists = {
            name: self._tokenize(hyp)
            for name, hyp in model_outputs.items()
        }

        # Align each model output against reference
        alignments: Dict[str, List[Tuple[Optional[str], Optional[str]]]] = {}
        for name, hyp_tokens in model_token_lists.items():
            alignments[name] = self._edit_align(ref_tokens, hyp_tokens)

        # Determine maximum alignment length across all models
        max_len = max((len(a) for a in alignments.values()), default=len(ref_tokens))

        # Pad alignments to max_len by appending (None, None)
        padded: Dict[str, List[Tuple]] = {}
        for name, aln in alignments.items():
            padded[name] = aln + [(None, None)] * (max_len - len(aln))

        # Build bins — iterate over aligned positions
        lattice: Lattice = []
        bin_idx = 0

        # Reconstruct ref-aligned bins
        # Strategy: iterate through ref positions; for each ref token,
        # create a bin and collect all model hypotheses at that aligned slot.
        ref_pos = 0
        for pos in range(max_len):
            # Gather what each model produced at this position
            model_hyps_at_pos: Dict[str, Optional[str]] = {}
            ref_word_at_pos: Optional[str] = None

            for name, aln in padded.items():
                if pos < len(aln):
                    ref_w, hyp_w = aln[pos]
                    if ref_word_at_pos is None and ref_w is not None:
                        ref_word_at_pos = ref_w
                    model_hyps_at_pos[name] = hyp_w
                else:
                    model_hyps_at_pos[name] = None

            if ref_word_at_pos is None:
                # Pure insertion from models — still create a bin
                lbin = LatticeBin(position=bin_idx)
                for hw in model_hyps_at_pos.values():
                    if hw:
                        for v in get_spelling_variants(hw):
                            lbin.add(v)
                if lbin.variants:
                    lattice.append(lbin)
                    bin_idx += 1
                continue

            lbin = LatticeBin(position=bin_idx)

            # Add reference word + all its variants
            for v in get_spelling_variants(ref_word_at_pos):
                lbin.add(v)

            # Add model hypotheses + variants
            valid_hyps = [h for h in model_hyps_at_pos.values() if h]
            for hw in valid_hyps:
                for v in get_spelling_variants(hw):
                    lbin.add(v)

            # Trust-model-over-reference heuristic:
            # If ≥ TRUST_THRESHOLD of models agree on a non-ref word, accept it.
            if valid_hyps:
                counts: Dict[str, int] = defaultdict(int)
                for hw in valid_hyps:
                    nw = unicodedata.normalize("NFC", hw.strip().lower())
                    counts[nw] += 1
                total = len(valid_hyps)
                for word_form, cnt in counts.items():
                    if cnt / total >= self.TRUST_THRESHOLD:
                        ref_norm = unicodedata.normalize("NFC",
                                        ref_word_at_pos.strip().lower())
                        if word_form != ref_norm:
                            lbin.add(word_form)
                            # Optionally log the override
                            # print(f"  [Trust override] pos={bin_idx}: ref={ref_word_at_pos} → {word_form}")

            lattice.append(lbin)
            bin_idx += 1

        return lattice


# ─────────────────────────────────────────────────────────────
# LATTICE-BASED WER COMPUTATION
# ─────────────────────────────────────────────────────────────

class LatticeWERComputer:
    """
    Computes WER for each model against the lattice.

    WER = (S + D + I) / N
    where:
      S = substitutions  (model word not in lattice bin)
      D = deletions       (model skipped a bin)
      I = insertions      (model produced extra words)
      N = number of bins in lattice (reference length)

    Algorithm:
      Dynamic programming over (lattice positions × hypothesis words),
      where at each position cost = 0 if hyp_word ∈ lattice_bin.variants,
      else cost = 1 (substitution). Insertion / deletion penalty = 1.
    """

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = unicodedata.normalize("NFC", text.strip())
        return [t for t in text.split() if t]

    def compute(self, lattice: Lattice, hypothesis: str) -> Dict:
        """
        Returns dict with keys: wer, substitutions, deletions, insertions,
        ref_length, hyp_length, alignment.
        """
        hyp_tokens = self._tokenize(hypothesis)
        L          = len(lattice)
        H          = len(hyp_tokens)

        if L == 0:
            return {"wer": 0.0, "substitutions": 0, "deletions": 0,
                    "insertions": H, "ref_length": 0, "hyp_length": H}

        # DP table: dp[i][j] = min edit cost for lattice[:i] vs hyp[:j]
        dp    = np.full((L + 1, H + 1), np.inf)
        dp[0][0] = 0
        for i in range(1, L + 1):
            dp[i][0] = i          # deletion cost
        for j in range(1, H + 1):
            dp[0][j] = j          # insertion cost

        for i in range(1, L + 1):
            for j in range(1, H + 1):
                # Match cost: 0 if hyp token is in lattice bin, else 1 (substitution)
                match_cost = 0 if lattice[i-1].matches(hyp_tokens[j-1]) else 1
                dp[i][j] = min(
                    dp[i-1][j-1] + match_cost,   # match / substitution
                    dp[i-1][j]   + 1,              # deletion
                    dp[i][j-1]   + 1,              # insertion
                )

        total_edits = int(dp[L][H])
        wer         = total_edits / L

        # Traceback to count S/D/I
        S = D = I = 0
        i, j = L, H
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                match_cost = 0 if lattice[i-1].matches(hyp_tokens[j-1]) else 1
                if dp[i][j] == dp[i-1][j-1] + match_cost:
                    if match_cost == 1:
                        S += 1
                    i -= 1; j -= 1
                    continue
            if i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1):
                D += 1; i -= 1
            else:
                I += 1; j -= 1

        return {
            "wer":           round(wer * 100, 2),
            "substitutions": S,
            "deletions":     D,
            "insertions":    I,
            "ref_length":    L,
            "hyp_length":    H,
            "total_errors":  total_edits,
        }

    def compute_standard_wer(self, reference: str, hypothesis: str) -> float:
        """Standard (non-lattice) WER for comparison."""
        ref_tokens = [unicodedata.normalize("NFC", w.lower())
                      for w in reference.split() if w]
        hyp_tokens = [unicodedata.normalize("NFC", w.lower())
                      for w in hypothesis.split() if w]
        R, H = len(ref_tokens), len(hyp_tokens)
        if R == 0:
            return float(H > 0)
        dp = np.arange(H + 1, dtype=float)
        for i in range(1, R + 1):
            prev = dp.copy()
            dp[0] = i
            for j in range(1, H + 1):
                cost = 0 if ref_tokens[i-1] == hyp_tokens[j-1] else 1
                dp[j] = min(prev[j-1] + cost, prev[j] + 1, dp[j-1] + 1)
        return round(dp[H] / R * 100, 2)


# ─────────────────────────────────────────────────────────────
# FULL EVALUATION DEMO
# ─────────────────────────────────────────────────────────────

def run_lattice_evaluation():
    """
    End-to-end demonstration using the book-buying sentence from the problem.
    """
    # ── Example from problem statement ──────────────────────
    reference = "उसने चौदह किताबें खरीदीं"

    model_outputs = {
        "Model_A": "उसने चौदह किताबें खरीदीं",           # perfect
        "Model_B": "उसने 14 किताबें खरीदी",              # digit, spelling variant
        "Model_C": "उसने चौदह पुस्तकें खरीदी",            # synonym पुस्तकें, spelling
        "Model_D": "उसने चौदह किताबे खरीदी",              # minor spelling variation
        "Model_E": "उसने चौदह किताबें खरीद",              # truncated output
    }

    print("\n" + "═" * 70)
    print("  LATTICE-BASED WER EVALUATION — DEMO")
    print("═" * 70)
    print(f'\n  Reference: "{reference}"')
    print(f"\n  Model outputs:")
    for name, out in model_outputs.items():
        print(f'    {name}: "{out}"')

    # Build lattice
    builder = LatticeBuilder()
    lattice = builder.build(reference, model_outputs)

    print(f"\n  Lattice ({len(lattice)} bins):")
    for i, lbin in enumerate(lattice):
        print(f"    Bin {i}: {sorted(lbin.variants)}")

    # Compute WER for each model
    computer = LatticeWERComputer()

    print(f"\n  {'Model':<12} {'Standard WER':>14} {'Lattice WER':>12} {'S':>4} {'D':>4} {'I':>4} {'Verdict'}")
    print("  " + "─" * 70)

    for name, hyp in model_outputs.items():
        std_wer     = computer.compute_standard_wer(reference, hyp)
        lattice_res = computer.compute(lattice, hyp)
        lat_wer     = lattice_res["wer"]
        improved    = "✓ fairer" if lat_wer < std_wer else ("=" if lat_wer == std_wer else "✗")
        print(f"  {name:<12} {std_wer:>13.1f}% {lat_wer:>11.1f}%"
              f"  {lattice_res['substitutions']:>3}  {lattice_res['deletions']:>3}"
              f"  {lattice_res['insertions']:>3}  {improved}")

    print()

    # ── Additional example — reference contains likely error ──
    print("\n  REFERENCE-ERROR HANDLING EXAMPLE")
    print("  " + "─" * 50)

    bad_reference = "उसने चौदाह किताबें खरीदीं"   # "चौदाह" is a typo for "चौदह"
    model_outputs_2 = {
        "Model_A": "उसने चौदह किताबें खरीदीं",
        "Model_B": "उसने चौदह किताबें खरीदीं",
        "Model_C": "उसने चौदह किताबें खरीदीं",
        "Model_D": "उसने चौदाह किताबें खरीदी",    # follows bad reference
    }
    print(f'  Bad reference: "{bad_reference}"')
    lattice_2 = builder.build(bad_reference, model_outputs_2)
    print(f"  Lattice Bin 1 (should contain 'चौदह' via trust override):")
    print(f"    {sorted(lattice_2[1].variants) if len(lattice_2) > 1 else 'N/A'}")

    print("\n  Comparing standard vs lattice WER for Model_A:")
    std   = computer.compute_standard_wer(bad_reference, model_outputs_2["Model_A"])
    lat   = computer.compute(lattice_2, model_outputs_2["Model_A"])
    print(f"    Standard WER = {std}%  (unfairly penalises correct output)")
    print(f"    Lattice WER  = {lat['wer']}%  (correctly accepts चौदह via model trust)")

    # Save live artifact report
    artifact = {
        "reference": reference,
        "models": model_outputs,
        "results": {
            name: {
                "standard_wer": computer.compute_standard_wer(reference, hyp),
                "lattice_wer": computer.compute(lattice, hyp)["wer"],
            }
            for name, hyp in model_outputs.items()
        },
        "trust_threshold": builder.TRUST_THRESHOLD,
        "alignment_unit": "word",
    }
    out_dir = Path("./artifacts/q4")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
    print(f"\n  Live Q4 report saved to {out_path}")


def build_q4_live_report(input_json_path: str, output_dir: str = "./artifacts/q4") -> Dict:
    """
    Build lattice-WER report from a JSON file with shape:
    {
      "reference": "...",
      "models": {"Model_A": "...", ...}
    }
    """
    with open(input_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    reference = payload["reference"]
    models = payload["models"]

    builder = LatticeBuilder()
    computer = LatticeWERComputer()
    lattice = builder.build(reference, models)

    report = {
        "reference": reference,
        "models": models,
        "alignment_unit": "word",
        "trust_threshold": builder.TRUST_THRESHOLD,
        "results": {},
    }

    for name, hyp in models.items():
        standard_wer = computer.compute_standard_wer(reference, hyp)
        lattice_res = computer.compute(lattice, hyp)
        report["results"][name] = {
            "standard_wer": standard_wer,
            "lattice_wer": lattice_res["wer"],
            "substitutions": lattice_res["substitutions"],
            "deletions": lattice_res["deletions"],
            "insertions": lattice_res["insertions"],
            "improved": lattice_res["wer"] < standard_wer,
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


# ─────────────────────────────────────────────────────────────
# PSEUDOCODE SUMMARY (printed as documentation)
# ─────────────────────────────────────────────────────────────

PSEUDOCODE = """
═══════════════════════════════════════════════════════════════════════
ALGORITHM SUMMARY — LATTICE-BASED WER
═══════════════════════════════════════════════════════════════════════

INPUT:
  reference R (string)
  model_outputs M = {m1: hyp1, m2: hyp2, ..., mN: hypN}

STEP 1 — TOKENISE
  ref_tokens   ← tokenise(R)
  hyp_tokens_i ← tokenise(hyp_i)   for each model i

STEP 2 — ALIGN EACH HYPOTHESIS TO REFERENCE (word-level N-W)
  alignment_i ← NeedlemanWunsch(ref_tokens, hyp_tokens_i)

STEP 3 — BUILD LATTICE BINS
  for each aligned position p:
    bin_p.variants ← {}
    bin_p.variants ← bin_p.variants ∪ spelling_variants(ref_token_at_p)
    for each model i:
      hw ← hyp_word_at_position(alignment_i, p)
      bin_p.variants ← bin_p.variants ∪ spelling_variants(hw)
    // Trust override
    model_agreement ← most_common(hyp_words_at_p)
    if agreement_fraction(model_agreement) ≥ 0.6 AND
       model_agreement ≠ ref_token_at_p:
        bin_p.variants.add(model_agreement)

STEP 4 — VARIANT EXPANSION (per word)
  spelling_variants(w):
    return {w, NFC(w), anusvara_swap(w), nukta_remove(w),
            deva_digit_ascii(w), number_word_digit(w), synonyms(w)}

STEP 5 — LATTICE WER (DP)
  dp[i][j] = min edit cost for lattice[:i] vs hyp[:j]
  cost(lattice_bin_i, hyp_word_j) = 0 if hyp_word_j ∈ bin_i.variants
                                   else 1
  dp[i][j] = min(dp[i-1][j-1] + cost(i,j),   # match/sub
                 dp[i-1][j]   + 1,             # deletion
                 dp[i][j-1]   + 1)             # insertion
  WER_model_k = dp[L][H] / L × 100%

OUTPUT:
  For each model k:
    lattice_WER_k  (reduces WER if model was unfairly penalised)
    standard_WER_k (for comparison)
    S, D, I counts
═══════════════════════════════════════════════════════════════════════
"""


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(PSEUDOCODE)
    run_lattice_evaluation()
