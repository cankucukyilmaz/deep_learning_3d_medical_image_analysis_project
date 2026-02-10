import os
import shutil
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_TRAIN_DIR = Path('data/raw/training')
RAW_TEST_DIR = Path('data/raw/testing')  # <--- Added
PROCESSED_DIR = Path('data/processed/Dataset500_ACDC')

def reset_processed_dir():
    """Wipes the processed directory to ensure reproducibility."""
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    
    (PROCESSED_DIR / 'imagesTr').mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / 'labelsTr').mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / 'imagesTs').mkdir(parents=True, exist_ok=True) # <--- Added
    print(f"Cleaned and created: {PROCESSED_DIR}")

def get_patient_group(patient_path):
    """Reads Info.cfg to find the disease group."""
    cfg_file = patient_path / 'Info.cfg'
    try:
        with open(cfg_file, 'r') as f:
            for line in f:
                if line.startswith('Group:'):
                    return line.split(':')[1].strip()
    except Exception:
        return 'Unknown'
    return 'Unknown'

def create_dataset():
    reset_processed_dir()
    
    # --- PART 1: TRAINING DATA (Patients 001-100) ---
    patient_ids = []
    groups = []
    case_identifiers = [] 
    
    train_folders = sorted([p for p in RAW_TRAIN_DIR.iterdir() if p.is_dir() and 'patient' in p.name])
    print(f"Processing {len(train_folders)} TRAINING patients...")

    for patient_path in tqdm(train_folders, desc="Training"):
        p_id = patient_path.name
        group = get_patient_group(patient_path)
        
        patient_ids.append(p_id)
        groups.append(group)
        
        # In Training, we look for Ground Truths to identify valid frames
        gt_files = sorted(list(patient_path.glob('*_gt.nii.gz')))
        
        for gt_file in gt_files:
            frame_id = gt_file.name.replace('_gt.nii.gz', '') 
            img_file = patient_path / (frame_id + '.nii.gz')
            
            if not img_file.exists(): continue
                
            case_identifiers.append(frame_id)
            
            # Create Symlinks
            dst_img = PROCESSED_DIR / 'imagesTr' / f"{frame_id}_0000.nii.gz"
            dst_lbl = PROCESSED_DIR / 'labelsTr' / f"{frame_id}.nii.gz"
            
            os.symlink(img_file.absolute(), dst_img)
            os.symlink(gt_file.absolute(), dst_lbl)

    # --- PART 2: TESTING DATA (Patients 101-150) ---
    # The test set has NO ground truth files (*_gt.nii.gz).
    # We must identify frames by looking for the images directly.
    # Usually, file is 'patient101_frame01.nii.gz'
    test_folders = sorted([p for p in RAW_TEST_DIR.iterdir() if p.is_dir() and 'patient' in p.name])
    print(f"Processing {len(test_folders)} TESTING patients...")

    for patient_path in tqdm(test_folders, desc="Testing"):
        # Find all .nii.gz files that do NOT end in _gt.nii.gz (just in case)
        # and are 4D or 3D? ACDC usually provides specific frames for testing too.
        # We look for the pattern 'patientXXX_frameXX.nii.gz'
        images = sorted(list(patient_path.glob('patient*_frame*.nii.gz')))
        
        for img_file in images:
            # Skip if it is a GT file (unlikely in test, but good safety)
            if '_gt' in img_file.name: continue
            
            frame_id = img_file.name.replace('.nii.gz', '')
            
            # Destination: imagesTs (Test Source)
            dst_img = PROCESSED_DIR / 'imagesTs' / f"{frame_id}_0000.nii.gz"
            
            os.symlink(img_file.absolute(), dst_img)

    # --- PART 3: STRATIFIED SPLITS (Training Only) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = []
    
    print("\nGenerating 5-Fold Stratified Splits (Training Data)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, groups)):
        train_patients = [patient_ids[i] for i in train_idx]
        val_patients = [patient_ids[i] for i in val_idx]
        
        train_cases = [c for c in case_identifiers if any(c.startswith(p) for p in train_patients)]
        val_cases = [c for c in case_identifiers if any(c.startswith(p) for p in val_patients)]
        
        splits.append({'train': train_cases, 'val': val_cases})

    # --- SAVE JSONS ---
    with open(PROCESSED_DIR / 'splits_final.json', 'w') as f:
        json.dump(splits, f, indent=4)

    dataset_json = {
        "channel_names": {"0": "MRI"}, 
        "labels": {"background": 0, "RV": 1, "Myocardium": 2, "LV": 3}, 
        "numTraining": len(case_identifiers), 
        "file_ending": ".nii.gz"
    }
    with open(PROCESSED_DIR / 'dataset.json', 'w') as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\nSuccess! Training and Test data linked in {PROCESSED_DIR}")

if __name__ == "__main__":
    create_dataset()