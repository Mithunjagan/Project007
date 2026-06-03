import json
from pathlib import Path

def inspect():
    audit_path = Path("evaluation/reports/pose_pipeline_audit.json")
    with open(audit_path, "r") as f:
        data = json.load(f)
        
    print(json.dumps(data, indent=2))
    
if __name__ == "__main__":
    inspect()
