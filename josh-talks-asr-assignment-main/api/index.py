from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json, sys, os
import traceback

# Add parent dir to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    from q2_cleanup_pipeline import ASRCleanupPipeline, HindiNumberNormalizer, EnglishWordDetector
    from q3_spell_checker import HindiSpellChecker, Verdict, Confidence
    from q4_lattice_wer import LatticeBuilder, LatticeWERComputer, get_spelling_variants
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Warning: Could not import dependencies: {e}")
    traceback.print_exc()

app = Flask(__name__)
CORS(app)

ARTIFACTS_DIR = os.path.join(parent_dir, "artifacts")


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ─────────────────────────────────────────────────────────────
# STATIC SERVING
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    html_path = os.path.join(parent_dir, "dashboard.html")
    if os.path.exists(html_path):
        return send_from_directory(parent_dir, "dashboard.html")
    return jsonify({"message": "Josh Talks ASR Assignment - Dashboard"}), 200


@app.route("/dashboard")
def dashboard():
    html_path = os.path.join(parent_dir, "dashboard.html")
    if os.path.exists(html_path):
        return send_from_directory(parent_dir, "dashboard.html")
    return jsonify({"message": "Dashboard"}), 200


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static assets (images, CSS, JS, etc.)"""
    try:
        return send_from_directory(parent_dir, filename)
    except Exception as e:
        return jsonify({"error": f"File not found: {filename}"}), 404


# Lazy load singletons to handle missing dependencies
pipeline = None
spell_check = None
lat_builder = None
lat_wer = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        try:
            pipeline = ASRCleanupPipeline()
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
    return pipeline

def get_spell_check():
    global spell_check
    if spell_check is None:
        try:
            spell_check = HindiSpellChecker()
        except Exception as e:
            print(f"Error initializing spell checker: {e}")
    return spell_check

def get_lat_builder():
    global lat_builder
    if lat_builder is None:
        try:
            lat_builder = LatticeBuilder()
        except Exception as e:
            print(f"Error initializing lattice builder: {e}")
    return lat_builder

def get_lat_wer():
    global lat_wer
    if lat_wer is None:
        try:
            lat_wer = LatticeWERComputer()
        except Exception as e:
            print(f"Error initializing lattice WER: {e}")
    return lat_wer


# ─────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Josh Talks ASR Demo API is running"}), 200


# ─────────────────────────────────────────────────────────────
# Q2 — NUMBER NORMALIZATION
# ─────────────────────────────────────────────────────────────

@app.route("/api/normalize", methods=["POST"])
def normalize():
    """
    POST { "text": "तीन सौ चौवन लोग आए" }
    → { "original": "...", "normalized": "354 लोग आए" }
    """
    try:
        data     = request.get_json(force=True)
        text     = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        pipeline_instance = get_pipeline()
        if pipeline_instance is None:
            return jsonify({"error": "Pipeline not available"}), 500
            
        result   = pipeline_instance.process(text)
        return jsonify({
            "original":   result.original,
            "normalized": result.normalized,
            "tagged":     result.tagged,
            "english_words": result.english_words,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# Q2 — ENGLISH WORD DETECTION
# ─────────────────────────────────────────────────────────────

@app.route("/api/detect-english", methods=["POST"])
def detect_english():
    """
    POST { "text": "मेरा interview कल है" }
    → { "tagged": "मेरा [EN]interview[/EN] कल है", "english_words": ["interview"] }
    """
    try:
        data  = request.get_json(force=True)
        text  = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        detector = EnglishWordDetector()
        tagged, eng_words = detector.detect(text)
        return jsonify({"original": text, "tagged": tagged, "english_words": eng_words}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# Q3 — SPELL CHECKER
# ─────────────────────────────────────────────────────────────

@app.route("/api/spell-check", methods=["POST"])
def spell_check_endpoint():
    """
    POST { "words": ["नमस्ते", "भारत्", "computर"] }
    → { "results": [{ word, verdict, confidence, reason }, ...] }
    """
    try:
        data  = request.get_json(force=True)
        words = data.get("words", [])
        if not words:
            return jsonify({"error": "words array is required"}), 400

        spell_check_instance = get_spell_check()
        if spell_check_instance is None:
            return jsonify({"error": "Spell checker not available"}), 500

        results = []
        for word in words[:500]:   # cap at 500 per request
            r = spell_check_instance.classify(word)
            results.append({
                "word":       r.word,
                "verdict":    r.verdict.value,
                "confidence": r.confidence.value,
                "reason":     r.reason,
            })

        correct_count   = sum(1 for r in results if r["verdict"] == "correct spelling")
        incorrect_count = len(results) - correct_count

        return jsonify({
            "results":         results,
            "total":           len(results),
            "correct_count":   correct_count,
            "incorrect_count": incorrect_count,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# Q4 — LATTICE WER
# ─────────────────────────────────────────────────────────────

@app.route("/api/lattice-wer", methods=["POST"])
def lattice_wer_endpoint():
    """
    POST {
      "reference": "उसने चौदह किताबें खरीदीं",
      "models": {
        "Model_A": "उसने चौदह किताबें खरीदीं",
        "Model_B": "उसने 14 किताबें खरीदी"
      }
    }
    """
    try:
        data      = request.get_json(force=True)
        reference = data.get("reference", "").strip()
        models    = data.get("models", {})

        if not reference or not models:
            return jsonify({"error": "reference and models are required"}), 400

        lat_builder_instance = get_lat_builder()
        lat_wer_instance = get_lat_wer()
        
        if lat_builder_instance is None or lat_wer_instance is None:
            return jsonify({"error": "Lattice WER not available"}), 500

        try:
            lattice = lat_builder_instance.build(reference, models)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        lattice_data = [{"position": b.position, "variants": sorted(b.variants)}
                        for b in lattice]

        wer_results = {}
        for name, hyp in models.items():
            std_wer = lat_wer_instance.compute_standard_wer(reference, hyp)
            lat_res = lat_wer_instance.compute(lattice, hyp)
            wer_results[name] = {
                "standard_wer":  std_wer,
                "lattice_wer":   lat_res["wer"],
                "substitutions": lat_res["substitutions"],
                "deletions":     lat_res["deletions"],
                "insertions":    lat_res["insertions"],
                "improved":      lat_res["wer"] < std_wer,
            }

        return jsonify({
            "reference":    reference,
            "lattice":      lattice_data,
            "wer_results":  wer_results,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# Q1 — WER TABLE (static results from training run)
# ─────────────────────────────────────────────────────────────

@app.route("/api/wer-table")
def wer_table():
    """Returns live Q1 evaluation results from artifacts/q1/report.json."""
    report_path = os.path.join(ARTIFACTS_DIR, "q1", "report.json")
    report = _read_json(report_path)
    if report is None:
        return jsonify({
            "error": "Q1 report artifact not found",
            "expected_path": report_path,
            "hint": "Run q1_whisper_finetune.py with a valid manifest to generate live metrics."
        }), 404
    return jsonify(report), 200


@app.route("/api/report-status")
def report_status():
    paths = {
        "q1_report": os.path.join(ARTIFACTS_DIR, "q1", "report.json"),
        "q2_report": os.path.join(ARTIFACTS_DIR, "q2", "report.json"),
        "q3_report": os.path.join(ARTIFACTS_DIR, "q3", "report.json"),
        "q4_report": os.path.join(ARTIFACTS_DIR, "q4", "report.json"),
    }
    return jsonify({
        "artifacts_dir": ARTIFACTS_DIR,
        "available": {name: os.path.exists(path) for name, path in paths.items()},
        "paths": paths,
    }), 200


@app.route("/api/q2-report")
def q2_report():
    report_path = os.path.join(ARTIFACTS_DIR, "q2", "report.json")
    report = _read_json(report_path)
    if report is None:
        return jsonify({
            "error": "Q2 report artifact not found",
            "expected_path": report_path,
            "hint": "Generate raw ASR pairs and run build_q2_live_report()."
        }), 404
    return jsonify(report), 200


@app.route("/api/q3-report")
def q3_report():
    report_path = os.path.join(ARTIFACTS_DIR, "q3", "report.json")
    report = _read_json(report_path)
    if report is None:
        return jsonify({
            "error": "Q3 report artifact not found",
            "expected_path": report_path,
            "hint": "Run spell check export and build_q3_live_report()."
        }), 404
    return jsonify(report), 200


@app.route("/api/q4-report")
def q4_report():
    report_path = os.path.join(ARTIFACTS_DIR, "q4", "report.json")
    report = _read_json(report_path)
    if report is None:
        return jsonify({
            "error": "Q4 report artifact not found",
            "expected_path": report_path,
            "hint": "Run q4_lattice_wer.py or build_q4_live_report() with model outputs input."
        }), 404
    return jsonify(report), 200


# Error handler
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found", "path": request.path}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "method": request.method}), 405


# Export app for Vercel
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
