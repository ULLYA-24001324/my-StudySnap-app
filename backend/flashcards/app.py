import json
import os
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK = boto3.client("bedrock-runtime", region_name="ap-southeast-1")
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

PROMPT = """You are a helpful study assistant. A student has provided the following raw lecture notes:

[Lecture Notes]

Based on these notes, generate exactly 5 flashcards covering the key concepts. Format each flashcard like this:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🃏 FLASHCARD 1 of 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📖 HOW TO USE: Read the question, think of your answer, then scroll down to check!

❓ QUESTION:
Write the question here

⬇️ Think of your answer first, then reveal below...
· · · · · · · · · · · · · · · ·
💡 ANSWER:
Write the answer here
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Repeat this format for all 5 flashcards, numbering them 1 of 5 through 5 of 5. Replace all placeholder text with real content from the lecture notes. Cover a broad range of key topics. Keep answers clear and concise."""


def build_messages(body: dict) -> list:
    file_data = body.get("file_data")
    file_mime = body.get("file_mime", "application/pdf")
    notes_text = body.get("notes", "")

    if file_data:
        if file_mime.startswith("image/"):
            content = [
                {"type": "image", "source": {"type": "base64", "media_type": file_mime, "data": file_data}},
                {"type": "text", "text": PROMPT},
            ]
        else:
            content = [
                {"type": "document", "source": {"type": "base64", "media_type": file_mime, "data": file_data}},
                {"type": "text", "text": PROMPT},
            ]
    else:
        content = [{"type": "text", "text": f"{PROMPT}\n\nNotes:\n{notes_text}"}]

    return [{"role": "user", "content": content}]


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@app.route("/", methods=["POST"])
def handler():
    body = request.get_json(force=True, silent=True) or {}
    messages = build_messages(body)

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": messages,
    }

    def generate():
        resp = BEDROCK.invoke_model_with_response_stream(
            modelId=MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        for event in resp["body"]:
            chunk = event.get("chunk")
            if chunk:
                data = json.loads(chunk["bytes"].decode())
                if data.get("type") == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")

    return Response(
        stream_with_context(generate()),
        content_type="text/plain; charset=utf-8",
        headers=CORS_HEADERS,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
