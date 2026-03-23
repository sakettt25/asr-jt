import json

def handler(request):
    """Health check endpoint"""
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
        },
        "body": json.dumps({
            "status": "ok",
            "message": "Josh Talks ASR Demo API is running"
        }),
    }
