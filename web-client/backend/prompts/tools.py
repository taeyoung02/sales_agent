"""
Tool definitions for LLM function calling
상세한 사용 가이드라인은 prompts/full/tool_guidelines.py 참조
"""

TOOL_REGISTRY = {
    "searchVehicleDatabase": {
        "definition": {
            "type": "function",
            "function": {
                "name": "searchVehicleDatabase",
                "description": "Search the vehicle database (RAG) for vehicle information, specifications, options, prices, and inventory. Use when user asks about vehicle details or needs database information. Do NOT use for 3D visualization requests (use generateHeatmap instead) or simple greetings. IMPORTANT: For VIN lookup, pass ONLY the VIN string (e.g., query='WBAU6D6U6R3R15LMG'), do NOT add words like '차량' or 'vehicle'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query. For VIN lookup, use ONLY the VIN string itself (e.g., query='WBAU6D6U6R3R15LMG'). For natural language search, use key terms from user's request (e.g., query='BMW iX3', query='3000만원대').",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    },
    "loadVehicle": {
        "definition": {
            "type": "function",
            "function": {
                "name": "loadVehicle",
                "description": "Load one or two vehicles in 3D viewers. vehicle_id MUST be VIN (17 characters) or 'test'. For comparison, use compare_with_vehicle_id parameter. source_path is automatically generated - do NOT provide it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vehicle_id": {
                            "type": "string",
                            "description": "Primary vehicle VIN (17 alphanumeric characters) or 'test'. VIN can be found in searchVehicleDatabase results metadata ('vin' field).",
                            "pattern": "^[A-Za-z0-9]{17}$|^test$",
                        },
                        "compare_with_vehicle_id": {
                            "type": "string",
                            "description": "Optional: Second vehicle VIN for side-by-side comparison. Use when user asks to compare two vehicles. Both vehicles will be loaded in separate 3D viewers.",
                            "pattern": "^[A-Za-z0-9]{17}$|^test$",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["ply", "gltf", "glb", "ksplat"],
                            "description": "3D asset format (optional, defaults to 'ply')",
                        },
                    },
                    "required": ["vehicle_id"],
                },
            },
        },
    },
    "generateHeatmap": {
        "definition": {
            "type": "function",
            "function": {
                "name": "generateHeatmap",
                "description": "Generate heatmap visualization to highlight a specific car part in the 3D model. **MUST USE** when user asks to 'show', 'see', 'find', 'highlight', 'zoom in', or 'examine' a specific part. Examples: '바퀴를 자세히 보여줘', '헤드라이트 보여줘', '사이드 미러 보여줘', 'wheel', 'headlight'. Works for external parts (wheel, headlight, bumper, door).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Car part or feature name to highlight. Extract only the part name from user's request. Examples: 'wheel' from '바퀴를 자세히 보여줘', 'trunk' from '트렁크를 보여줘', 'trunk' from '트렁크 공간 보여줘', 'headlight' from '헤드라이트 보여줘'.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
    },
    "setCamera": {
        "definition": {
            "type": "function",
            "function": {
                "name": "setCamera",
                "description": "Set camera preset or custom position for 3D viewer. Use when user asks to change camera view (e.g., '앞면 보여줘', '옆면 보여줘', '위에서 보여줘'). Do NOT use together with generateHeatmap - generateHeatmap automatically moves the camera.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "preset": {
                            "type": "string",
                            "enum": [
                                "initial",
                                "front",
                                "back",
                                "left",
                                "right",
                                "top",
                                "front_windshield",
                                "rear_windshield",
                            ],
                            "description": "Camera preset position. Use 'initial' for first loaded view, 'front_windshield' for elevated front view, 'rear_windshield' for elevated rear view.",
                        },
                        "position": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "Custom camera position [x, y, z]. Use only if preset doesn't meet your needs.",
                        },
                    },
                    "required": [],
                },
            },
        },
    },
    "generatePresentationScript": {
        "definition": {
            "type": "function",
            "function": {
                "name": "generatePresentationScript",
                "description": "Generate TTS script with timestamped 3D viewer controls for automated vehicle presentation. Use when user asks to introduce, explain, or present a vehicle. CRITICAL: Must call loadVehicle first. If vehicle_id is an actual VIN, then call searchVehicleDatabase(query=vehicle_id) with VIN string exactly (no additional words), and use ONLY database results to generate script. If vehicle_id is 'test', DO NOT call searchVehicleDatabase and DO NOT try to map it to a real VIN; generate a generic demo script instead. Always include 'duration' field in all camera actions (setCamera: 2-3s, zoomCamera: 1-2s, rotateCamera: 2-4s).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vehicle_id": {
                            "type": "string",
                            "description": "Vehicle identifier to present. If this is a real vehicle, it MUST be a VIN (17 characters) for which you have already called searchVehicleDatabase(query=vehicle_id) with the VIN string exactly (no additional words) to retrieve vehicle information. If this is the special demo vehicle 'test', DO NOT call searchVehicleDatabase for it and DO NOT try to map it to a real VIN; generate a reasonable generic explanation instead.",
                        },
                        "script": {
                            "type": "object",
                            "description": "Presentation script with segments. Generate using ONLY data from searchVehicleDatabase results. Include duration field in all camera actions (setCamera: 2-3s, zoomCamera: 1-2s, rotateCamera: 2-4s).",
                            "properties": {
                                "total_duration": {
                                    "type": "number",
                                    "description": "Total presentation duration in seconds",
                                },
                                "segments": {
                                    "type": "array",
                                    "description": "Array of presentation segments with timestamp, text, and actions",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "timestamp": {
                                                "type": "number",
                                                "description": "Segment start time in seconds",
                                            },
                                            "text": {
                                                "type": "string",
                                                "description": "Narration text for TTS. Use natural conversational Korean. Base content ONLY on searchVehicleDatabase results.",
                                            },
                                            "actions": {
                                                "type": "array",
                                                "description": "Camera actions synchronized with narration. Always include 'duration' field for smooth animations.",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "type": {
                                                            "type": "string",
                                                            "enum": [
                                                                "setCamera",
                                                                "zoomCamera",
                                                                "rotateCamera",
                                                                "generateHeatmap",
                                                            ],
                                                            "description": "Action type",
                                                        },
                                                        "preset": {
                                                            "type": "string",
                                                            "description": "Camera preset (for setCamera): initial, front, back, left, right, top, front_windshield, rear_windshield",
                                                        },
                                                        "position": {
                                                            "type": "array",
                                                            "items": {"type": "number"},
                                                            "description": "Custom camera position [x, y, z] (for setCamera, rarely needed)",
                                                        },
                                                        "level": {
                                                            "type": "number",
                                                            "description": "Zoom level 0-100 (for zoomCamera): 40-60 for wide/medium, 80-100 for close-up",
                                                        },
                                                        "direction": {
                                                            "type": "string",
                                                            "enum": ["in", "out"],
                                                            "description": "Zoom direction (for zoomCamera, alternative to level)",
                                                        },
                                                        "target": {
                                                            "type": "array",
                                                            "items": {"type": "number"},
                                                            "description": "Target point [x, y, z] to look at (for rotateCamera)",
                                                        },
                                                        "query": {
                                                            "type": "string",
                                                            "description": "Car part name for heatmap (for generateHeatmap)",
                                                        },
                                                        "duration": {
                                                            "type": "number",
                                                            "description": "Animation duration in seconds. REQUIRED for all actions: setCamera (2.0-3.0s), zoomCamera (1.0-2.0s), rotateCamera (2.0-4.0s)",
                                                        },
                                                        "delay": {
                                                            "type": "number",
                                                            "description": "Delay after segment timestamp before executing action (optional)",
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                        "required": [
                                            "timestamp",
                                            "text",
                                            "actions",
                                        ],
                                    },
                                },
                            },
                            "required": ["segments"],
                        },
                    },
                    "required": ["vehicle_id", "script"],
                },
            },
        },
    },
}
