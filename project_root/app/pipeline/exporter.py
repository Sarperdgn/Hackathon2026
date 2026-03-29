import json

def export_json(output_path, payload):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)