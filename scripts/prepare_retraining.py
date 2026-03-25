# scripts/prepare_retraining.py
import os
import json
from pathlib import Path
from PIL import Image

# --- Configuration ---
COLLECTED_DIR = Path("data/collected/labeled")
STATUS_REPORT = Path("reports/pipeline_status.json")
RETRAIN_THRESHOLD = 50  # Minimum new images required to trigger retraining

def validate_and_count():
    """Scan labeled folders and check image integrity."""
    report = {
        "status": "pending",
        "total_new_images": 0,
        "classes": {},
        "is_ready_for_retraining": False
    }
    
    classes = ["fire", "smoke", "non fire"]
    
    for cls in classes:
        cls_path = COLLECTED_DIR / cls
        if not cls_path.exists():
            report["classes"][cls] = 0
            continue
            
        # Count only valid images
        valid_files = 0
        for img_path in cls_path.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    valid_files += 1
                except:
                    print(f"Removing corrupt image: {img_path}")
                    os.remove(img_path)
                    
        report["classes"][cls] = valid_files
        report["total_new_images"] += valid_files

    # Check if threshold is met
    if report["total_new_images"] >= RETRAIN_THRESHOLD:
        report["status"] = "ready"
        report["is_ready_for_retraining"] = True
    
    return report

def main():
    print("🚀 Starting data pipeline status check...")
    
    # 1. Analyze collected data
    report_data = validate_and_count()
    
    # 2. Save report to JSON
    os.makedirs(STATUS_REPORT.parent, exist_ok=True)
    with open(STATUS_REPORT, "w") as f:
        json.dump(report_data, f, indent=4)
        
    print(f"📊 Report generated: {STATUS_REPORT}")
    print(f"Total images collected: {report_data['total_new_images']}/{RETRAIN_THRESHOLD}")
    
    if report_data["is_ready_for_retraining"]:
        print("✅ SUCCESS: Enough data collected for retraining!")
    else:
        print("⏳ Still collecting more data...")

if __name__ == "__main__":
    main()