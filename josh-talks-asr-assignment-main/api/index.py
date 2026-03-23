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
            import json
            import sys
            import os

            # Add parent dir to path
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, parent_dir)

            ARTIFACTS_DIR = os.path.join(parent_dir, "artifacts")
            def _read_json(path):
                if not os.path.exists(path):
                    return None
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except:
                    return None
            def handler(request):
                path = request.path
    
                # Root
                if path in ('/', ''):
                    dashboard_path = os.path.join(parent_dir, 'dashboard.html')
                    if os.path.exists(dashboard_path):
                        with open(dashboard_path, 'r', encoding='utf-8') as f:
                            return {
                                "statusCode": 200,
                                "headers": {"Content-Type": "text/html; charset=utf-8"},
                                "body": f.read(),
                            }
                    return {
                        "statusCode": 200,
                        "headers": {"Content-Type": "application/json"},
                        "body": json.dumps({"status": "OK", "message": "Josh Talks ASR API"}),
                    }
                # Health check
                if path == '/api/health':
                    return {
                        "statusCode": 200,
                        "headers": {"Content-Type": "application/json"},
                        "body": json.dumps({"status": "ok", "message": "API running"}),
                    }
    
                # Report endpoints
                if path == '/api/wer-table':
                    report = _read_json(os.path.join(ARTIFACTS_DIR, "q1", "report.json"))
                    return {
                        "statusCode": 200 if report else 404,
                        "headers": {"Content-Type": "application/json"},
                        "body": json.dumps(report or {"error": "Not found"}),
                    }
    
                if path == '/api/report-status':
                    return {
                        "statusCode": 200,
                        "headers": {"Content-Type": "application/json"},
                        "body": json.dumps({
                            "available": {
                                "q1": os.path.exists(os.path.join(ARTIFACTS_DIR, "q1", "report.json")),
                                "q2": os.path.exists(os.path.join(ARTIFACTS_DIR, "q2", "report.json")),
                                "q3": os.path.exists(os.path.join(ARTIFACTS_DIR, "q3", "report.json")),
                                "q4": os.path.exists(os.path.join(ARTIFACTS_DIR, "q4", "report.json")),
                            }
                        }),
                    }
    
                if path == '/api/q2-report':
                    report = _read_json(os.path.join(ARTIFACTS_DIR, "q2", "report.json"))
                    return {
                        "statusCode": 200 if report else 404,
                        "headers": {"Content-Type": "application/json"},
                        "body": json.dumps(report or {"error": "Not found"}),
                    }
    
                if path == '/api/q3-report':
                    report = _read_json(os.path.join(ARTIFACTS_DIR, "q3", "report.json"))
                    return {
                        "statusCode": 200 if report else 404,
                        "headers": {"Content-Type": "application/json"},
                        "body": json.dumps(report or {"error": "Not found"}),
                    }
    
                if path == '/api/q4-report':
                    report = _read_json(os.path.join(ARTIFACTS_DIR, "q4", "report.json"))
                    return {
                        "statusCode": 200 if report else 404,
                        "headers": {"Content-Type": "application/json"},
                        "body": json.dumps(report or {"error": "Not found"}),
                    }
                # 404
                return {
                    "statusCode": 404,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"error": "Not found", "path": path}),
                }
def spell_check_endpoint():
