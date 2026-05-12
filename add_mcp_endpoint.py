#!/usr/bin/env python3
"""
Run this from your project root:  python add_mcp_endpoint.py
This adds a proper MCP JSON-RPC 2.0 endpoint that Prompt Opinion can connect to.
"""
import os, shutil, sys

MCP_CODE = r'''
# ==================== PROPER MCP SERVER ENDPOINT ====================
# Prompt Opinion requires JSON-RPC 2.0 over HTTP (Streamable HTTP transport)
# Single endpoint at /mcp that handles initialize, tools/list, tools/call

import json as _json

_MCP_TOOLS = [
    {
        "name": "traffic_light_detector",
        "description": "Detects traffic light color (red/green/yellow) from a base64 image. RED = safe to cross, GREEN = stop.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array with one base64 encoded JPEG image of the traffic light"
                }
            },
            "required": ["frames"]
        }
    },
    {
        "name": "food_identifier",
        "description": "Identifies food items visible in an image. Returns food names, quantities, and meal type.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of base64 encoded JPEG images showing food"
                }
            },
            "required": ["frames"]
        }
    },
    {
        "name": "document_reader",
        "description": "Reads and extracts all text from a document, book page, prescription, or bill in an image.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_data": {
                    "type": "string",
                    "description": "Base64 encoded JPEG image of the document"
                }
            },
            "required": ["image_data"]
        }
    },
    {
        "name": "scene_describer",
        "description": "Describes what is in front of a visually impaired person. Specifies object locations (left, right, near, far).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of base64 encoded JPEG images of the scene"
                },
                "question": {
                    "type": "string",
                    "description": "Optional question about the scene"
                }
            },
            "required": ["frames"]
        }
    }
]


def _mcp_call_tool(name: str, arguments: dict) -> str:
    """Route tool calls to existing vision functions."""
    try:
        if name == "traffic_light_detector":
            frames = arguments.get("frames", [])
            if not frames:
                return "No image provided."
            processed = [preprocess_image(f, 800) for f in frames[:1] if f and len(f) > 100]
            if not processed:
                return "Could not process image."
            return groq_vision_call(TRAFFIC_SYSTEM, TRAFFIC_USER, processed, max_tokens=80, priority="high")

        elif name == "food_identifier":
            frames = arguments.get("frames", [])
            if not frames:
                return "No image provided."
            processed = [preprocess_image(f, 800) for f in frames[:4] if f and len(f) > 100]
            if not processed:
                return "Could not process image."
            return groq_vision_call(FOOD_SYSTEM, FOOD_USER, processed, max_tokens=250, priority="low")

        elif name == "document_reader":
            image_data = arguments.get("image_data", "")
            if not image_data and arguments.get("frames"):
                image_data = arguments["frames"][0]
            if not image_data:
                return "No image provided."
            processed = preprocess_image(image_data, 900)
            if not processed:
                return "Could not process image."
            return groq_vision_call(PAGE_READER_SYSTEM, PAGE_READER_USER, [processed], max_tokens=1500, priority="high")

        elif name == "scene_describer":
            frames = arguments.get("frames", [])
            question = arguments.get("question", "What is in front of me? Describe the scene.")
            if not frames:
                return "No image provided."
            processed = [preprocess_image(f, 900) for f in frames[:3] if f and len(f) > 100]
            if not processed:
                return "Could not process image."
            user_prompt = (
                f'The user asks: "{question}". '
                'Describe clearly what you see for a visually impaired person. '
                'Be specific about locations (left, right, center, near, far).'
            )
            return groq_vision_call(
                "You are describing a scene for a visually impaired person. Be clear and specific. No markdown. Natural speech.",
                user_prompt, processed, max_tokens=300, priority="high"
            )
        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Error running {name}: {str(e)}"


@app.route('/mcp', methods=['GET', 'POST', 'DELETE', 'OPTIONS'])
def mcp_endpoint():
    """
    Proper MCP JSON-RPC 2.0 endpoint (Streamable HTTP transport).
    Prompt Opinion connects here.
    """
    # CORS headers for Prompt Opinion
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, Mcp-Session-Id, MCP-Protocol-Version',
        'Content-Type': 'application/json'
    }

    if request.method == 'OPTIONS':
        return '', 204, headers

    if request.method == 'GET':
        # SSE stream for server-initiated messages — return 405 (we don't need it)
        return jsonify({"error": "SSE not supported, use POST"}), 405, headers

    if request.method == 'DELETE':
        return '', 200, headers

    # POST — handle JSON-RPC
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({
            "jsonrpc": "2.0", "id": None,
            "error": {"code": -32700, "message": "Parse error"}
        }), 400, headers

    rpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {})

    # --- initialize ---
    if method == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "VisionAssist AI Smart Glasses",
                    "version": "1.0.0"
                }
            }
        }
        return jsonify(response), 200, headers

    # --- initialized (notification, no response needed) ---
    if method == "notifications/initialized":
        return '', 202, headers

    # --- tools/list ---
    if method == "tools/list":
        response = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {"tools": _MCP_TOOLS}
        }
        return jsonify(response), 200, headers

    # --- tools/call ---
    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result_text = _mcp_call_tool(tool_name, arguments)
        response = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": False
            }
        }
        return jsonify(response), 200, headers

    # --- ping ---
    if method == "ping":
        return jsonify({"jsonrpc": "2.0", "id": rpc_id, "result": {}}), 200, headers

    # --- unknown method ---
    return jsonify({
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"}
    }), 200, headers


@app.route('/mcp/health', methods=['GET'])
def mcp_proper_health():
    return jsonify({"status": "healthy", "version": "1.0", "tools": [t["name"] for t in _MCP_TOOLS]})

'''

MARKER = 'if __name__ == "__main__":'

def patch(filepath='app.py'):
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found. Run this from your project root.")
        sys.exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'def mcp_endpoint()' in content:
        print("MCP endpoint already exists in app.py — no changes made.")
        return

    idx = content.rfind(MARKER)
    if idx == -1:
        print(f"ERROR: Could not find '{MARKER}' in app.py")
        sys.exit(1)

    shutil.copy(filepath, filepath + '.backup')
    print(f"Backup saved -> {filepath}.backup")

    new_content = content[:idx] + MCP_CODE + "\n\n" + content[idx:]
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("\nSUCCESS: Proper MCP endpoint added to app.py")
    print("\nMCP server URL to use in Prompt Opinion:")
    print("  https://competion-blind-deployed.onrender.com/mcp")
    print("\nNext steps:")
    print("  1. git add app.py")
    print("  2. git commit -m 'Add proper MCP JSON-RPC endpoint'")
    print("  3. git push")
    print("  4. In Prompt Opinion -> MCP Servers:")
    print("     Endpoint: https://competion-blind-deployed.onrender.com/mcp")
    print("     Transport Type: Streamable HTTP  (or HTTP)")
    print("     Auth: None")

if __name__ == '__main__':
    patch()