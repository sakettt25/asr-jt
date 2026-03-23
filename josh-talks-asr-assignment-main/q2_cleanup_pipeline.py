"""
Question 2: ASR Cleanup Pipeline — Number Normalization + English Word Detection
Josh Talks | AI Researcher Intern - Speech & Audio Assignment

This module builds a two-stage post-processing pipeline for raw Hindi ASR output:
  a) Number Normalization  — spoken Hindi number words → Arabic digits
  b) English Word Detection — tag Roman/English tokens in Devanagari transcripts
"""

import re, json, requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

MODEL_NAME  = "openai/whisper-small"
LANGUAGE    = "hindi"
TASK        = "transcribe"
SAMPLE_RATE = 16_000

# ─────────────────────────────────────────────────────────────
# PART A — NUMBER NORMALIZATION
# ─────────────────────────────────────────────────────────────

# Ones
ONES = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "पांच": 5, "छः": 6, "छह": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23,
    "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27,
    "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30, "इकतीस": 31,
    "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
    "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43,
    "चवालीस": 44, "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47,
    "अड़तालीस": 48, "उनचास": 49, "पचास": 50,
    "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54,
    "पचपन": 55, "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58,
    "उनसठ": 59, "साठ": 60, "इकसठ": 61, "बासठ": 62,
    "तिरसठ": 63, "चौंसठ": 64, "पैंसठ": 65, "छियासठ": 66,
    "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69, "सत्तर": 70,
    "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74,
    "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78,
    "उन्यासी": 79, "अस्सी": 80, "इक्यासी": 81, "बयासी": 82,
    "तिरासी": 83, "चौरासी": 84, "पचासी": 85, "छियासी": 86,
    "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89, "नब्बे": 90,
    "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94,
    "पचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98,
    "निन्यानवे": 99,
}

MULTIPLIERS = {
    "सौ":    100,
    "हज़ार":  1_000,
    "हजार":  1_000,
    "लाख":   100_000,
    "करोड़":  10_000_000,
    "करोड":  10_000_000,
    "अरब":   1_000_000_000,
}

# Idioms / fixed phrases where we should NOT convert numbers
IDIOM_PHRASES = {
    "दो-चार",       # "a few" — idiom
    "दो चार",
    "चार-छह",
    "दो तीन",
    "दो-तीन",
    "तीन-चार",
    "पाँच-छह",
    "एक-दो",
    "सात-आठ",
    "नौ-दस",
    "चार पाँच",
    "दस-पंद्रह",     # "ten to fifteen" used idiomatically
    "एक न एक",       # "one or the other"
    "दो टूक",        # "bluntly"
    "तीन तेरह",      # "confusion / scattered" (idiom)
    "चार चाँद",      # "adding glory" (idiom)
    "सात समुद्र",    # "seven seas" (idiom)
    "नौ दो ग्यारह",  # "ran away" (idiom)
    "दस बीस",
}

# Devanagari digit characters → ASCII
DEVA_DIGIT_MAP = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}


class HindiNumberNormalizer:
    """
    Converts spoken Hindi number words to Arabic digit strings.

    Algorithm:
      1. Pre-scan the text for known idiom phrases and protect them
         with placeholders so they are not altered.
      2. Tokenise by whitespace, then use a greedy left-to-right pass
         through the token list.
      3. Accumulate tokens that can form a valid number expression
         (ones + optional multiplier chain).
      4. When accumulation stalls, emit the digit string and reset.
      5. Restore idiom placeholders.
      6. Convert Devanagari digit characters (०-९) to ASCII (0-9).

    Edge-case handling:
      - Ordinals (पहली, दूसरा, तीसरा) are NOT converted (they are
        not cardinal numbers and converting would change meaning).
      - Time expressions like "तीन बजे" are converted ("3 बजे")
        because the digit form is standard.
      - Idiomatic phrases (see IDIOM_PHRASES) are protected.
    """

    # Ordinal words — leave as-is
    ORDINALS = {
        "पहला", "पहली", "पहले", "दूसरा", "दूसरी", "दूसरे",
        "तीसरा", "तीसरी", "तीसरे", "चौथा", "पाँचवाँ", "छठा",
        "सातवाँ", "आठवाँ", "नौवाँ", "दसवाँ",
    }

    def __init__(self):
        self._ones  = ONES
        self._mults = MULTIPLIERS

    # ── helpers ──────────────────────────────────────────────

    def _protect_idioms(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replace idiom phrases with placeholders."""
        placeholders = {}
        protected    = text
        for phrase in sorted(IDIOM_PHRASES, key=len, reverse=True):
            if phrase in protected:
                key = f"__IDIOM{len(placeholders)}__"
                placeholders[key] = phrase
                protected = protected.replace(phrase, key)
        return protected, placeholders

    def _restore_idioms(self, text: str, placeholders: Dict[str, str]) -> str:
        for key, val in placeholders.items():
            text = text.replace(key, val)
        return text

    def _deva_to_ascii_digits(self, text: str) -> str:
        for deva, ascii_d in DEVA_DIGIT_MAP.items():
            text = text.replace(deva, ascii_d)
        return text

    def _tokens_to_number(self, tokens: List[str]) -> Optional[int]:
        """
        Greedy parse: try to interpret a list of Hindi word tokens
        as a single integer value.  Returns None if not a valid number.
        """
        total = 0
        current = 0

        for token in tokens:
            if token in self.ORDINALS:
                return None                 # ordinals don't convert
            if token in self._ones:
                current += self._ones[token]
            elif token in self._mults:
                mult = self._mults[token]
                if current == 0:
                    current = 1             # "सौ" alone → 100
                if mult >= 1000:
                    total  = (total + current) * mult
                    current = 0
                else:
                    current *= mult         # "तीन सौ" → 300
            else:
                return None

        return total + current if (total + current) > 0 else None

    def normalize(self, text: str) -> str:
        """Main entry point: normalise numbers in a Hindi string."""
        # Step 1: Protect Devanagari digits → convert first (simple sub)
        text = self._deva_to_ascii_digits(text)

        # Step 2: Protect idiom phrases
        text, placeholders = self._protect_idioms(text)

        # Step 3: Greedy token accumulation
        tokens = text.split()
        output = []
        i = 0

        while i < len(tokens):
            # Try to greedily accumulate a number span
            best_end    = -1
            best_number = None

            for j in range(i + 1, min(i + 7, len(tokens) + 1)):
                span   = tokens[i:j]
                number = self._tokens_to_number(span)
                if number is not None:
                    best_end    = j
                    best_number = number

            if best_number is not None:
                output.append(str(best_number))
                i = best_end
            else:
                output.append(tokens[i])
                i += 1

        result = " ".join(output)

        # Step 4: Restore idioms
        result = self._restore_idioms(result, placeholders)
        return result


# ─────────────────────────────────────────────────────────────
# PART A — BEFORE/AFTER EXAMPLES
# ─────────────────────────────────────────────────────────────

NORMALIZER_EXAMPLES = [
    # (input, expected_output, note)
    ("मुझे दो सौ रुपये चाहिए",       "मुझे 200 रुपये चाहिए",    "Simple compound: दो सौ → 200"),
    ("तीन सौ चौवन लोग आए",           "354 लोग आए",              "Three-part: तीन सौ चौवन → 354"),
    ("एक हज़ार पाँच सौ",              "1500",                    "Thousands + hundreds"),
    ("पच्चीस प्रतिशत छूट मिली",       "25 प्रतिशत छूट मिली",    "Simple: पच्चीस → 25"),
    ("दो लाख रुपये का लोन",           "200000 रुपये का लोन",     "Lakh multiplier"),
    # Edge cases
    ("दो-चार बातें करते हैं",         "दो-चार बातें करते हैं",  "EDGE: idiom — should stay as-is"),
    ("नौ दो ग्यारह हो गया",           "नौ दो ग्यारह हो गया",    "EDGE: idiom 'ran away'"),
    ("वो पहली बार आया था",            "वो पहली बार आया था",      "EDGE: ordinal पहली — no conversion"),
]


def demo_normalizer():
    n = HindiNumberNormalizer()
    print("\n══ NUMBER NORMALIZATION — Before / After Examples ══\n")
    print(f"{'Input':<40} {'Output':<35} {'Note'}")
    print("─" * 120)
    for inp, expected, note in NORMALIZER_EXAMPLES:
        result = n.normalize(inp)
        status = "✓" if result == expected else "✗"
        print(f"{status}  {inp:<40} {result:<35} {note}")
    print()


# ─────────────────────────────────────────────────────────────
# PART B — ENGLISH WORD DETECTION
# ─────────────────────────────────────────────────────────────

class EnglishWordDetector:
    """
    Identifies English (Roman-script) tokens in a Hindi transcript
    and wraps them in [EN]...[/EN] tags.

    Detection strategy (multi-signal):
      1. Script check  — if a token has >50% Latin Unicode characters
         it is almost certainly an English/Roman word.
      2. Devanagari transliteration of loanwords  — some English words
         appear as Devanagari (e.g., "इंटरव्यू"). These are already
         correctly handled by the transcription guideline (Devanagari
         Devanagari IS correct; we do NOT tag these).
      3. Mixed-script tokens  — tokens like "IT" or "AI" in Roman
         within Hindi text are tagged.
      4. Common abbreviations / acronyms  — short uppercase ASCII
         strings (2–5 chars) are tagged as English.

    What we do NOT tag:
      - Devanagari-script loanwords (following the transcription
        guideline: "कंप्यूटर" is correct, not an error).
      - Numbers / digits (handled by the normalizer).
      - Punctuation.
    """

    # High-confidence Devanagari loanword forms — these are CORRECT per guidelines
    # and should NOT be tagged as English.
    DEVA_LOANWORDS = {
        "इंटरव्यू", "कंप्यूटर", "मोबाइल", "फ़ोन", "ऑनलाइन",
        "ऑफलाइन", "क्लास", "कॉलेज", "स्कूल", "जॉब",
        "वर्क", "टीम", "मैनेजर", "कंपनी", "ऑफिस",
        "बिज़नेस", "प्रॉब्लम", "सॉल्यूशन", "सक्सेस",
        "यूट्यूब", "व्हाट्सएप", "फेसबुक", "इंस्टाग्राम",
        "वीडियो", "पॉडकास्ट", "स्टार्टअप", "इंटर्नशिप",
    }

    @staticmethod
    def _is_roman_token(token: str) -> bool:
        """Return True if token is predominantly Latin-script."""
        if not token:
            return False
        alpha_chars = [c for c in token if c.isalpha()]
        if not alpha_chars:
            return False
        latin_count = sum(1 for c in alpha_chars if ord(c) < 256)
        return (latin_count / len(alpha_chars)) > 0.5

    @staticmethod
    def _is_devanagari_token(token: str) -> bool:
        deva = sum(1 for c in token if "\u0900" <= c <= "\u097f")
        return (deva / max(len(token), 1)) > 0.3

    @staticmethod
    def _is_number(token: str) -> bool:
        return bool(re.match(r"^\d+([.,]\d+)?$", token))

    @staticmethod
    def _is_punctuation(token: str) -> bool:
        return all(not c.isalnum() for c in token)

    def detect(self, text: str) -> Tuple[str, List[str]]:
        """
        Returns:
          tagged_text   — original text with [EN]...[/EN] markers
          english_words — list of detected English words
        """
        tokens        = re.split(r"(\s+)", text)   # preserve whitespace
        tagged_parts  = []
        english_words = []

        for token in tokens:
            if token.strip() == "" or self._is_punctuation(token.strip()):
                tagged_parts.append(token)
            elif self._is_number(token.strip()):
                tagged_parts.append(token)
            elif token.strip() in self.DEVA_LOANWORDS:
                # Correct Devanagari loanword — do NOT tag
                tagged_parts.append(token)
            elif self._is_roman_token(token.strip()):
                tagged_parts.append(f"[EN]{token.strip()}[/EN]")
                english_words.append(token.strip())
                # Preserve trailing space if any
                # (already handled by split on whitespace)
            else:
                tagged_parts.append(token)

        tagged_text = "".join(tagged_parts)
        return tagged_text, english_words


# ─────────────────────────────────────────────────────────────
# PART B — BEFORE/AFTER EXAMPLES
# ─────────────────────────────────────────────────────────────

DETECTION_EXAMPLES = [
    "मेरा interview कल है और मुझे nervous लग रहा है",
    "यह problem solve नहीं हो रहा है मुझे help चाहिए",
    "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",      # Deva loanwords: no tag
    "उसने YouTube पर video देखा और inspired हो गया",
    "हमारी company का target achieve हो गया है",
    "AI और machine learning का बहुत scope है आजकल",
    "मेरा CGPA 8.5 है और मैं IT sector में जाना चाहता हूँ",
]


def demo_detector():
    d = EnglishWordDetector()
    print("\n══ ENGLISH WORD DETECTION — Before / After Examples ══\n")
    for sentence in DETECTION_EXAMPLES:
        tagged, eng_words = d.detect(sentence)
        print(f"  Input  : {sentence}")
        print(f"  Output : {tagged}")
        print(f"  EN tags: {eng_words}")
        print()


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    original:      str
    normalized:    str          # after number normalization
    tagged:        str          # after English word detection
    english_words: List[str] = field(default_factory=list)


class ASRCleanupPipeline:
    """
    Two-stage post-processing pipeline:
      Stage 1: Number Normalization
      Stage 2: English Word Detection

    Usage:
      pipeline = ASRCleanupPipeline()
      result   = pipeline.process("मुझे तीन सौ रुपये और interview की date चाहिए")
    """

    def __init__(self):
        self.normalizer = HindiNumberNormalizer()
        self.detector   = EnglishWordDetector()

    def process(self, raw_asr: str) -> PipelineResult:
        normalized          = self.normalizer.normalize(raw_asr)
        tagged, eng_words   = self.detector.detect(normalized)
        return PipelineResult(
            original=raw_asr,
            normalized=normalized,
            tagged=tagged,
            english_words=eng_words,
        )

    def process_batch(self, texts: List[str]) -> List[PipelineResult]:
        return [self.process(t) for t in texts]

    def evaluate_on_dataset(self, pairs: List[Tuple[str, str]]) -> Dict:
        """
        pairs: list of (raw_asr_output, human_reference) tuples.
        Computes WER before and after pipeline application.
        """
        try:
            import evaluate as hf_eval
            wer_metric = hf_eval.load("wer")
        except ImportError:
            print("Install 'evaluate' package for WER computation.")
            return {}

        raws, refs, processed = [], [], []
        for raw, ref in pairs:
            result = self.process(raw)
            # For WER eval strip tags (they are not in reference)
            clean_pred = result.tagged.replace("[EN]", "").replace("[/EN]", "")
            raws.append(raw)
            refs.append(ref)
            processed.append(clean_pred)

        before_wer = 100 * wer_metric.compute(predictions=raws,      references=refs)
        after_wer  = 100 * wer_metric.compute(predictions=processed, references=refs)

        return {
            "before_wer": round(before_wer, 2),
            "after_wer":  round(after_wer, 2),
            "delta":      round(before_wer - after_wer, 2),
            "n_samples":  len(pairs),
        }


def build_q2_live_report(
    pairs: List[Tuple[str, str]],
    output_dir: str = "./artifacts/q2",
    max_examples: int = 5,
) -> Dict:
    """
    Build a live Q2 report from actual ASR/reference pairs.
    Writes artifacts/q2/report.json and returns the report dict.
    """
    pipeline = ASRCleanupPipeline()
    results = [pipeline.process(raw) for raw, _ in pairs]

    conversions = []
    edge_cases = []
    english_tag_examples = []

    for result in results:
        if result.original != result.normalized and len(conversions) < max_examples:
            conversions.append({
                "input": result.original,
                "output": result.normalized,
            })

        if any(phrase in result.original for phrase in IDIOM_PHRASES) and len(edge_cases) < 3:
            edge_cases.append({
                "input": result.original,
                "output": result.normalized,
                "note": "Idiom detected; preserved where conversion would alter meaning",
            })

        if result.english_words and len(english_tag_examples) < max_examples:
            english_tag_examples.append({
                "input": result.normalized,
                "tagged_output": result.tagged,
                "english_words": result.english_words,
            })

    wer_summary = pipeline.evaluate_on_dataset(pairs) if pairs else {}

    report = {
        "n_pairs": len(pairs),
        "wer_summary": wer_summary,
        "number_normalization": {
            "converted_examples": conversions,
            "edge_case_examples": edge_cases,
        },
        "english_detection": {
            "tagged_examples": english_tag_examples,
            "loanword_lexicon_size": len(pipeline.detector.DEVA_LOANWORDS),
        },
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    return report


# ─────────────────────────────────────────────────────────────
# GENERATE ASR TRANSCRIPTS (using pretrained Whisper-small)
# for the ~10-hour dataset (called from Q1 data)
# ─────────────────────────────────────────────────────────────

def generate_asr_transcripts(manifest_records: List[Dict],
                              output_path: str = "raw_asr_pairs.json") -> List[Tuple[str, str]]:
    """
    Run pretrained whisper-small on each audio segment in the manifest
    to generate raw ASR transcripts. Pairs each with the human reference.

    Returns list of (raw_asr, reference) tuples.
    """
    import torch, numpy as np, torchaudio
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    pairs = []
    cache = Path("./audio_cache")
    cache.mkdir(exist_ok=True)

    for rec in manifest_records:
        rid = rec["recording_id"]
        # Fetch reference
        try:
            ref_data = requests.get(rec["transcription_url"], timeout=15).json()
            reference = ref_data.get("transcription") or ref_data.get("text", "")
        except Exception:
            continue

        # Load audio
        cache_path = cache / f"{rid}.wav"
        if not cache_path.exists():
            try:
                r = requests.get(rec["rec_url_gcp"], timeout=30)
                cache_path.write_bytes(r.content)
            except Exception:
                continue
        try:
            waveform, sr = torchaudio.load(str(cache_path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
            audio = waveform.squeeze().numpy().astype(np.float32)
        except Exception:
            continue

        # Run inference
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(
                inputs.input_features.to(device),
                language=LANGUAGE, task=TASK, num_beams=1
            )
        raw_asr = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        pairs.append((raw_asr, reference.strip()))

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([{"raw_asr": p, "reference": r} for p, r in pairs], f,
                  ensure_ascii=False, indent=2)

    print(f"Generated {len(pairs)} ASR pairs → {output_path}")
    return pairs


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo the normalizer
    demo_normalizer()

    # Demo the detector
    demo_detector()

    # Full pipeline demo
    pipeline = ASRCleanupPipeline()
    test_inputs = [
        "मुझे तीन सौ रुपये और interview की date बताओ",
        "पच्चीस लोगों ने apply किया लेकिन दो-चार को ही select करेंगे",
        "एक हज़ार पाँच सौ students ने online course join किया",
        "वो नौ दो ग्यारह हो गया जब manager ने performance review लिया",
    ]

    print("\n══ FULL PIPELINE — COMBINED OUTPUT ══\n")
    for text in test_inputs:
        result = pipeline.process(text)
        print(f"  Original  : {result.original}")
        print(f"  Normalized: {result.normalized}")
        print(f"  Tagged    : {result.tagged}")
        print(f"  EN words  : {result.english_words}")
        print()

    # Optional live report generation if raw ASR pairs file exists.
    pairs_file = Path("raw_asr_pairs.json")
    if pairs_file.exists():
        with open(pairs_file, "r", encoding="utf-8") as handle:
            items = json.load(handle)
        live_pairs = [(item["raw_asr"], item.get("reference", "")) for item in items]
        report = build_q2_live_report(live_pairs, output_dir="./artifacts/q2")
        print("Live Q2 report generated at ./artifacts/q2/report.json")
        print(f"Pairs analyzed: {report['n_pairs']}")
