from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import sys
import os
import traceback

# Add parent dir to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Initialize Flask app
app = Flask(__name__, static_folder=parent_dir, static_url_path='')
CORS(app)

# Configure artifacts directory
ARTIFACTS_DIR = os.path.join(parent_dir, "artifacts")

# Try to import dependencies
try:
    from q2_cleanup_pipeline import ASRCleanupPipeline, EnglishWordDetector
    from q3_spell_checker import HindiSpellChecker
    from q4_lattice_wer import LatticeBuilder, LatticeWERComputer
    HAS_DEPS = True
except ImportError as e:
    print(f"Warning: Dependencies not available: {e}")
    HAS_DEPS = False
    traceback.print_exc()


def _read_json(path: str):
    """Read JSON file safely"""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# ROOT ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def root():
    """Serve dashboard.html"""
    dashboard_path = os.path.join(parent_dir, 'dashboard.html')
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    return jsonify({
        "message": "Josh Talks ASR Assignment API",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "normalize": "/api/normalize",
            "detect-english": "/api/detect-english",
            "spell-check": "/api/spell-check",
            "lattice-wer": "/api/lattice-wer",
            "wer-table": "/api/wer-table",
            "report-status": "/api/report-status"
        }
    }), 200


# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "Josh Talks ASR Demo API is running",
        "has_dependencies": HAS_DEPS,
        "artifacts_dir": ARTIFACTS_DIR
    }), 200


@app.route('/api/normalize', methods=['POST'])
def normalize():
    """Q2 - Number normalization endpoint"""
    try:
        if not HAS_DEPS:
            return jsonify({"error": "Dependencies not available"}), 503
            
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        pipeline = ASRCleanupPipeline()
        result = pipeline.process(text)
        return jsonify({
            "original": result.original,
            "normalized": result.normalized,
            "tagged": result.tagged,
            "english_words": result.english_words,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/detect-english', methods=['POST'])
def detect_english():
    """Q2 - English word detection endpoint"""
    try:
        if not HAS_DEPS:
            return jsonify({"error": "Dependencies not available"}), 503
            
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        detector = EnglishWordDetector()
        tagged, eng_words = detector.detect(text)
        return jsonify({
            "original": text,
            "tagged": tagged,
            "english_words": eng_words
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/spell-check', methods=['POST'])
def spell_check_endpoint():
    """Q3 - Spell checker endpoint"""
    try:
        if not HAS_DEPS:
            return jsonify({"error": "Dependencies not available"}), 503
            
        data = request.get_json(force=True)
        words = data.get("words", [])
        if not words:
            return jsonify({"error": "words array is required"}), 400

        spell_check = HindiSpellChecker()
        results = []
        for word in words[:500]:
            r = spell_check.classify(word)
            results.append({
                "word": r.word,
                "verdict": r.verdict.value,
                "confidence": r.confidence.value,
                "reason": r.reason,
            })

        correct_count = sum(1 for r in results if r["verdict"] == "correct spelling")
        incorrect_count = len(results) - correct_count

        return jsonify({
            "results": results,
            "total": len(results),
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/lattice-wer', methods=['POST'])
def lattice_wer_endpoint():
    """Q4 - Lattice WER endpoint"""
    try:
        if not HAS_DEPS:
            return jsonify({"error": "Dependencies not available"}), 503
            
        data = request.get_json(force=True)
        reference = data.get("reference", "").strip()
        models = data.get("models", {})

        if not reference or not models:
            return jsonify({"error": "reference and models are required"}), 400

        lat_builder = LatticeBuilder()
        lat_wer = LatticeWERComputer()

        lattice = lat_builder.build(reference, models)
        lattice_data = [{"position": b.position, "variants": sorted(b.variants)}
                        for b in lattice]

        wer_results = {}
        for name, hyp in models.items():
            std_wer = lat_wer.compute_standard_wer(reference, hyp)
            lat_res = lat_wer.compute(lattice, hyp)
            wer_results[name] = {
                "standard_wer": std_wer,
                "lattice_wer": lat_res["wer"],
                "substitutions": lat_res["substitutions"],
                "deletions": lat_res["deletions"],
                "insertions": lat_res["insertions"],
                "improved": lat_res["wer"] < std_wer,
            }

        return jsonify({
            "reference": reference,
            "lattice": lattice_data,
            "wer_results": wer_results,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# REPORT ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.route('/api/wer-table', methods=['GET'])
def wer_table():
    """Q1 - WER table from artifacts"""
    report_path = os.path.join(ARTIFACTS_DIR, "q1", "report.json")
    report = _read_json(report_path)
    if report is None:
        return jsonify({
            "error": "Q1 report artifact not found",
            "expected_path": report_path,
        }), 404
    return jsonify(report), 200


@app.route('/api/report-status', methods=['GET'])
def report_status():
    """Check which reports are available"""
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


@app.route('/api/q2-report', methods=['GET'])
def q2_report():
    """Q2 - Report from artifacts"""
    report_path = os.path.join(ARTIFACTS_DIR, "q2", "report.json")
    report = _read_json(report_path)
    if report is None:
        return jsonify({
            "error": "Q2 report artifact not found",
            "expected_path": report_path,
        }), 404
    return jsonify(report), 200


@app.route('/api/q3-report', methods=['GET'])
def q3_report():
    """Q3 - Report from artifacts"""
    report_path = os.path.join(ARTIFACTS_DIR, "q3", "report.json")
    report = _read_json(report_path)
    if report is None:
        return jsonify({
            "error": "Q3 report artifact not found",
            "expected_path": report_path,
        }), 404
    return jsonify(report), 200


@app.route('/api/q4-report', methods=['GET'])
def q4_report():
    """Q4 - Report from artifacts"""
    report_path = os.path.join(ARTIFACTS_DIR, "q4", "report.json")
    report = _read_json(report_path)
    if report is None:
        return jsonify({
            "error": "Q4 report artifact not found",
            "expected_path": report_path,
        }), 404
    return jsonify(report), 200


# ─────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    """404 error handler"""
    return jsonify({
        "error": "Not found",
        "path": request.path,
        "method": request.method
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """405 error handler"""
    return jsonify({
        "error": "Method not allowed",
        "path": request.path,
        "method": request.method
    }), 405


@app.errorhandler(500)
def internal_error(e):
    """500 error handler"""
    return jsonify({
        "error": "Internal server error",
        "message": str(e)
    }), 500


# Export app for Vercel
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
