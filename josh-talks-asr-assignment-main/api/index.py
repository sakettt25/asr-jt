from flask import Flask
from flask_cors import CORS
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

app = Flask(__name__)
CORS(app)

def read_json(path):
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return None

@app.route("/", methods=["GET"])
def index():
    try:
        html = os.path.join(BASE_DIR, "dashboard.html")
        if os.path.exists(html):
            with open(html) as f:
                return f.read(), 200
        return json.dumps({"status": "ok"}), 200
    except Exception as e:
        return json.dumps({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return json.dumps({"status": "ok"}), 200

@app.route("/api/wer-table", methods=["GET"])
def wer_table():
    data = read_json(os.path.join(ARTIFACTS_DIR, "q1", "report.json"))
    return (json.dumps(data), 200) if data else (json.dumps({"error": "not found"}), 404)

@app.route("/api/q2-report", methods=["GET"])
def q2_report():
    data = read_json(os.path.join(ARTIFACTS_DIR, "q2", "report.json"))
    return (json.dumps(data), 200) if data else (json.dumps({"error": "not found"}), 404)

@app.route("/api/q3-report", methods=["GET"])
def q3_report():
    data = read_json(os.path.join(ARTIFACTS_DIR, "q3", "report.json"))
    return (json.dumps(data), 200) if data else (json.dumps({"error": "not found"}), 404)

@app.route("/api/q4-report", methods=["GET"])
def q4_report():
    data = read_json(os.path.join(ARTIFACTS_DIR, "q4", "report.json"))
    return (json.dumps(data), 200) if data else (json.dumps({"error": "not found"}), 404)

@app.route("/api/report-status", methods=["GET"])
def report_status():
    return json.dumps({
        "q1": os.path.exists(os.path.join(ARTIFACTS_DIR, "q1", "report.json")),
        "q2": os.path.exists(os.path.join(ARTIFACTS_DIR, "q2", "report.json")),
        "q3": os.path.exists(os.path.join(ARTIFACTS_DIR, "q3", "report.json")),
        "q4": os.path.exists(os.path.join(ARTIFACTS_DIR, "q4", "report.json")),
    }), 200

if __name__ == "__main__":
    app.run(debug=False)
