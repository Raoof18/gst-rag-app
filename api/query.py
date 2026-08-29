import json
import sys
import os
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implementation.answer import answer_question


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
            question = body.get("question", "").strip()
            style = body.get("style", "precise")
            history = body.get("history", [])  # list of {role, text} from the client, oldest first

            if not question:
                self._respond(400, {"error": "Missing 'question' in request body"})
                return

            prev_exchange = self._extract_last_exchange(history)

            answer, sources = answer_question(question, style=style, prev_exchange=prev_exchange, conversation_history=history)

            self._respond(200, {"answer": answer, "sources": sources})

        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _extract_last_exchange(self, history):
        """history is oldest-first [{role, text}, ...]. Find the last bot message and the
        user message immediately before it -- that's the 'previous exchange' for follow-up context."""
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "bot":
                prev_answer = history[i].get("text", "")
                if i > 0 and history[i - 1].get("role") == "user":
                    return {"question": history[i - 1].get("text", ""), "answer": prev_answer}
        return None

    def _respond(self, status, data):
        self.send_response(status)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
