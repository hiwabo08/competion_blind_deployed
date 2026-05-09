#!/usr/bin/env python3
"""
Run this script from your project root to add MCP endpoints to app.py.
Usage:  python patch_app.py
"""
import os
import shutil
import sys

MCP_ROUTES = r'''
# ==================== MCP DIRECT ENDPOINTS (No camera check) ====================
# These endpoints are for the Prompt Opinion hackathon.
# They bypass the camera_active check and work directly with base64 images.

@app.route('/api/mcp/traffic', methods=['POST'])
def mcp_traffic():
    """MCP endpoint for traffic light detection - no camera check required"""
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        if not frames:
            return jsonify({"success": False, "result": "No image provided. Please send a base64 encoded image."})
        processed = [preprocess_image(f, 800) for f in frames[:1] if f and len(f) > 100]
        if not processed:
            return jsonify({"success": False, "result": "Could not process the image. Please ensure it's a valid JPEG."})
        result = groq_vision_call(TRAFFIC_SYSTEM, TRAFFIC_USER, processed, max_tokens=80, priority="high")
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "result": f"Error: {str(e)}"})


@app.route('/api/mcp/food', methods=['POST'])
def mcp_food():
    """MCP endpoint for food identification - no camera check required"""
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        if not frames:
            return jsonify({"success": False, "result": "No image provided. Please send a base64 encoded image."})
        processed = [preprocess_image(f, 800) for f in frames[:4] if f and len(f) > 100]
        if not processed:
            return jsonify({"success": False, "result": "Could not process the image. Please ensure it's a valid JPEG."})
        result = groq_vision_call(FOOD_SYSTEM, FOOD_USER, processed, max_tokens=250, priority="low")
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "result": f"Error: {str(e)}"})


@app.route('/api/mcp/read', methods=['POST'])
def mcp_read():
    """MCP endpoint for document/page reading - no camera check required"""
    try:
        data = request.get_json()
        image_data = data.get('image_data', '')
        # Also support 'frames' array for compatibility
        if not image_data and data.get('frames'):
            image_data = data.get('frames', [])[0] if data.get('frames') else ''
        if not image_data:
            return jsonify({"success": False, "result": "No image provided. Please send a base64 encoded image."})
        processed = preprocess_image(image_data, 900)
        if not processed:
            return jsonify({"success": False, "result": "Could not process the image. Please ensure it's a valid JPEG."})
        result = groq_vision_call(PAGE_READER_SYSTEM, PAGE_READER_USER, [processed], max_tokens=1500, priority="high")
        return jsonify({"success": True, "result": result, "description": result})
    except Exception as e:
        return jsonify({"success": False, "result": f"Error: {str(e)}"})


@app.route('/api/mcp/scene', methods=['POST'])
def mcp_scene():
    """MCP endpoint for scene description - no camera check required"""
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        question = data.get('question', 'What is in front of me? Describe the scene.')
        if not frames:
            return jsonify({"success": False, "result": "No image provided. Please send a base64 encoded image."})
        processed = [preprocess_image(f, 900) for f in frames[:3] if f and len(f) > 100]
        if not processed:
            return jsonify({"success": False, "result": "Could not process the image. Please ensure it's a valid JPEG."})
        user_prompt = (
            f'The user asks: "{question}". '
            'Describe clearly what you see in this image for a visually impaired person. '
            'Be specific about locations (left, right, center, near, far).'
        )
        result = groq_vision_call(
            "You are describing a scene for a visually impaired person. Be clear, helpful, and specific about object locations. No markdown. Natural speech.",
            user_prompt,
            processed,
            max_tokens=300,
            priority="high"
        )
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "result": f"Error: {str(e)}"})


@app.route('/api/mcp/health', methods=['GET'])
def mcp_health():
    """Health check endpoint for MCP server"""
    return jsonify({"status": "healthy", "version": "1.0", "tools": ["traffic", "food", "read", "scene"]})

'''

INSERTION_MARKER = 'if __name__ == "__main__":'


def patch_app(filepath='app.py'):
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found in current directory.")
        print("Make sure you are running this from your project root.")
        sys.exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'def mcp_health()' in content:
        print("MCP endpoints already exist in app.py — no changes made.")
        return

    idx = content.rfind(INSERTION_MARKER)
    if idx == -1:
        print(f"ERROR: Could not find '{INSERTION_MARKER}' in app.py")
        sys.exit(1)

    backup_path = filepath + '.backup'
    shutil.copy(filepath, backup_path)
    print(f"Backup saved -> {backup_path}")

    new_content = content[:idx] + MCP_ROUTES + "\n\n" + content[idx:]

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("\nSUCCESS: MCP endpoints added to app.py")
    print("\nNew endpoints after deploy:")
    print("  GET  /api/mcp/health")
    print("  POST /api/mcp/traffic  {'frames': ['base64...']}")
    print("  POST /api/mcp/food     {'frames': ['base64...']}")
    print("  POST /api/mcp/read     {'image_data': 'base64...'}")
    print("  POST /api/mcp/scene    {'frames': ['base64...'], 'question': 'optional'}")
    print("\nNext steps:")
    print("  1. git add app.py")
    print("  2. git commit -m 'Add MCP endpoints for hackathon'")
    print("  3. git push")
    print("  4. curl https://competion-blind-deployed.onrender.com/api/mcp/health")


if __name__ == '__main__':
    patch_app()