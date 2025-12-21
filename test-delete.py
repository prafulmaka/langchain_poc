import json
import logging
import os
from typing import Dict, Any

# --------------------------------------------------
# Logging
# --------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --------------------------------------------------
# Environment variables
# --------------------------------------------------
STAGE = os.getenv("STAGE", "dev")
APP_NAME = os.getenv("APP_NAME", "example-lambda")

# --------------------------------------------------
# Business logic
# --------------------------------------------------
def process_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core business logic lives here.
    Keeps lambda_handler thin and testable.
    """
    name = data.get("name", "World")
    return {
        "message": f"Hello {name}!",
        "stage": STAGE,
        "app": APP_NAME
    }

# --------------------------------------------------
# Lambda entry point
# --------------------------------------------------
def lambda_handler(event, context):
    logger.info("Lambda invoked")
    logger.info("Event: %s", json.dumps(event))

    try:
        # Example: direct invocation or test event
        body = event.get("body")

        if body and isinstance(body, str):
            body = json.loads(body)
        elif body is None:
            body = event

        result = process_request(body)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result)
        }

    except Exception as e:
        logger.exception("Unhandled exception")

        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "Internal Server Error",
                "message": str(e)
            })
        }