import json
import os
import random
import boto3

BEDROCK = boto3.client("bedrock-runtime", region_name="ap-southeast-1")
IMAGE_MODEL_ID = "amazon.nova-canvas-v1:0"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

IMAGE_PROMPT_TEMPLATE = (
    "Create an adorable, cute illustration representing the main topic and key concepts from these lecture notes: "
    "{notes}. Use a kawaii or chibi art style with soft pastel colors, friendly characters, and charming details. "
    "Include cute mascot-like characters or animals related to the subject matter, with big eyes and cheerful "
    "expressions. The overall feel should be warm, playful, and delightful while still being educational and "
    "relevant to the topic."
)


def handler(event, context):
    # Handle CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": "",
        }

    try:
        body = json.loads(event.get("body") or "{}")
    except (json.JSONDecodeError, TypeError):
        body = {}

    notes_text = body.get("notes", "general academic subject")
    # Truncate notes for image prompt — Nova Canvas has a 1024 char limit on prompts
    truncated = notes_text[:400] if len(notes_text) > 400 else notes_text
    image_prompt = IMAGE_PROMPT_TEMPLATE.format(notes=truncated)

    seed = random.randint(0, 858993459)

    payload = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": image_prompt,
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "width": 1280,
            "height": 720,
            "seed": seed,
            "quality": "standard",
            "cfgScale": 8.0,
        },
    }

    try:
        response = BEDROCK.invoke_model(
            modelId=IMAGE_MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        image_b64 = result["images"][0]

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"image": image_b64}),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
