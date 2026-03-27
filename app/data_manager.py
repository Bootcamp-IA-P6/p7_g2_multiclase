# app/data_manager.py
import os
import shutil
from pathlib import Path

# Use absolute paths or reliable relative paths
BASE_DIR = Path(__file__).parent.parent
COLLECTED_DIR = BASE_DIR / "data" / "collected"
PENDING_DIR = COLLECTED_DIR / "pending"
LABELED_DIR = COLLECTED_DIR / "labeled"

def init_folders():
    """Ensure all required folders exist."""
    # Create main directories
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    for cls in ["fire", "smoke", "non fire"]:
        (LABELED_DIR / cls).mkdir(parents=True, exist_ok=True)

def save_to_pending(uploaded_file, prediction_id):
    init_folders()
    ext = os.path.splitext(uploaded_file.name)[1]
    filename = f"{prediction_id}{ext}"
    save_path = PENDING_DIR / filename
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    print(f"📸 [MLOps] Image saved to pending: {save_path}")
    return filename

def move_to_labeled(prediction_id, target_label):
    """Finds the file by ID and moves it to the correct labeled folder."""
    init_folders()
    
    # 1. Clean the label name (remove spaces/lowercase to be safe)
    # Our folders are 'fire', 'smoke', 'non fire'
    target_label = target_label.lower().strip()
    
    # 2. Search for the file in pending
    found_file = None
    if not PENDING_DIR.exists():
        print(f"❌ [MLOps] Pending directory does not exist!")
        return False

    for f in os.listdir(PENDING_DIR):
        if f.startswith(f"{prediction_id}."):
            found_file = f
            break
    
    if not found_file:
        print(f"⚠️ [MLOps] Could not find file for ID {prediction_id} in pending.")
        return False

    # 3. Move the file
    source_path = PENDING_DIR / found_file
    target_path = LABELED_DIR / target_label / found_file
    
    try:
        shutil.move(str(source_path), str(target_path))
        print(f"✅ [MLOps] Moved {found_file} to {target_label} folder.")
        return True
    except Exception as e:
        print(f"❌ [MLOps] Error moving file: {e}")
        return False