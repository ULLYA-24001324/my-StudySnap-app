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

SYSTEM_PROMPT_TEMPLATE = """Generate 5 multiple-choice quiz questions based on the lecture notes provided here:

{lecture_notes}

Do not reveal the correct answers yet. You are a quiz master. Present one question at a time. Start by presenting Question 1 with 4 options (A, B, C, D) and wait for the user's answer before proceeding. After the user answers, tell them if they are correct or not, briefly explain the right answer, then move on to the next question. After all 5 questions, show a final score out of 5. Then ask the user to rate their confidence level on this topic from 1 to 5, where 1 means very unsure and 5 means very confident. Based on their rating, provide personalised study tips and recommendations."""


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@app.route("/", methods=["POST"])
def handler():
    body = request.get_json(force=True, silent=True) or {}

    # Support both direct file upload and pre-extracted text
    file_data = body.get("file_data")
    file_mime = body.get("file_mime", "application/pdf")
    notes_text = body.get("notes", "No lecture notes provided.")
    history = body.get("history", [])
    message = body.get("message", "")

    # Build system prompt with lecture notes context
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lecture_notes=notes_text)

    # Build messages list: history + new user message
    messages = list(history)  # copy existing history [{role, content}]

    # If there is a file upload and this is the first message (no history),
    # prepend file content to the first user message
    if file_data and not history:
        if file_mime.startswith("image/"):
            file_block = {"type": "image", "source": {"type": "base64", "media_type": file_mime, "data": file_data}}
        else:
            file_block = {"type": "document", "source": {"type": "base64", "media_type": file_mime, "data": file_data}}

        messages.append({
            "role": "user",
            "content": [
                file_block,
                {"type": "text", "text": message or "start"},
            ],
        })
    else:
        if message:
            messages.append({"role": "user", "content": message})

    if not messages:
        messages.append({"role": "user", "content": "start"})

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": system_prompt,
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
