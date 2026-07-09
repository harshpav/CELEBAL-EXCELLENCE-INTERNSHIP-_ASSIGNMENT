from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from agent import agent

app = Flask(__name__, static_folder="../frontend")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/agent", methods=["POST"])
def agent_endpoint():
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    response = agent(query)
    return jsonify(response)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Agent is running!"})


if __name__ == "__main__":
    print("🚀 Agent Project running at http://localhost:5000")
    app.run(debug=True, port=5000)
