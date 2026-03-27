# app/data_manager.py
import os
import shutil

# --- Configuration ---
BASE_COLLECTED_DIR = "data/collected"
PENDING_DIR = os.path.join(BASE_COLLECTED_DIR, "pending")
LABELED_DIR = os.path.join(BASE_COLLECTED_DIR, "labeled")

def init_folders():
    """Ensure all required folders exist."""
    for folder in [PENDING_DIR, 
                   os.path.join(LABELED_DIR, "fire"), 
                   os.path.join(LABELED_DIR, "smoke"), 
                   os.path.join(LABELED_DIR, "non fire")]:
        os.makedirs(folder, exist_ok=True)

def save_to_pending(uploaded_file, prediction_id):
    """
    Save file using the Supabase ID as the filename.
    Example: 123.jpg
    """
    init_folders()
    
    ext = os.path.splitext(uploaded_file.name)[1]
    # Use the DB ID as the unique filename
    unique_name = f"{prediction_id}{ext}"
    save_path = os.path.join(PENDING_DIR, unique_name)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return unique_name

def move_to_labeled(prediction_id, target_label):
    """
    Find the file starting with the ID and move it.
    """
    init_folders()
    # Search for the file in pending (since we don't know the exact extension)
    found_file = None
    for f in os.listdir(PENDING_DIR):
        if f.startswith(f"{prediction_id}."):
            found_file = f
            break
    
    if found_file:
        source_path = os.path.join(PENDING_DIR, found_file)
        target_path = os.path.join(LABELED_DIR, target_label, found_file)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.move(source_path, target_path)
        return True
    return False