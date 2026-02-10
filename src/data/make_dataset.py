import os
import shutil
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# --- CONFIGURATION ---
# We use relative paths assuming this runs from project root
RAW_DATA_DIR = Path('data/raw/training')
PROCESSED_DIR = Path('data/processed/Dataset500_ACDC') # ID 500 for nnU-Net custom dataset

def reset_processed_dir():
    """Wipes the processed directory to ensure reproducibility."""
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    
    (PROCESSED_DIR / 'imagesTr').mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / 'labelsTr').mkdir(parents=True, exist_ok=True)
    print(f"cleaned and created: {PROCESSED_DIR}")

def get_patient_group(patient_path):
    """Reads Info.cfg to find the disease group (NOR, MINF, DCM, HCM, RV)."""
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
    
    patient_ids = []
    groups = []
    case_identifiers = [] # Stores 'patient001_frame01' etc.
    
    # Sort to ensure deterministic order
    patient_folders = sorted([p for p in RAW_DATA_DIR.iterdir() if p.is_dir() and 'patient' in p.name])
    
    print(f"Structing data for {len(patient_folders)} patients...")

    for patient_path in tqdm(patient_folders):
        p_id = patient_path.name
        group = get_patient_group(patient_path)
        
        patient_ids.append(p_id)
        groups.append(group)
        
        # Find Ground Truths (e.g., patient001_frame01_gt.nii.gz)
        gt_files = sorted(list(patient_path.glob('*_gt.nii.gz')))
        
        for gt_file in gt_files:
            # 1. Identify filenames
            frame_id = gt_file.name.replace('_gt.nii.gz', '') # e.g. patient001_frame01
            img_file = patient_path / (frame_id + '.nii.gz')
            
            if not img_file.exists():
                print(f"Missing image for {gt_file}")
                continue
                
            case_identifiers.append(frame_id)
            
            # 2. Create Symlinks (The "Memory Efficient" part)
            # Source (Real file)
            src_img = img_file.absolute()
            src_lbl = gt_file.absolute()
            
            # Destination (Virtual link for nnU-Net)
            # nnU-Net needs images to end in _0000.nii.gz
            dst_img = PROCESSED_DIR / 'imagesTr' / f"{frame_id}_0000.nii.gz"
            dst_lbl = PROCESSED_DIR / 'labelsTr' / f"{frame_id}.nii.gz"
            
            try:
                os.symlink(src_img, dst_img)
                os.symlink(src_lbl, dst_lbl)
            except FileExistsError:
                pass # Should not happen due to reset, but safe to ignore

    # --- STRATIFIED SPLITTING ---
    # We split patients, then map back to cases (frames)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = []
    
    print("\nGenerating 5-Fold Stratified Splits...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, groups)):
        train_patients = [patient_ids[i] for i in train_idx]
        val_patients = [patient_ids[i] for i in val_idx]
        
        # Convert Patient IDs to Case IDs (Frame IDs)
        # Check if the case string starts with the patient string
        train_cases = [c for c in case_identifiers if any(c.startswith(p) for p in train_patients)]
        val_cases = [c for c in case_identifiers if any(c.startswith(p) for p in val_patients)]
        
        splits.append({'train': train_cases, 'val': val_cases})
        
        # Verify Balance
        val_groups = [groups[i] for i in val_idx]
        unique, counts = np.unique(val_groups, return_counts=True)
        print(f"Fold {fold}: {dict(zip(unique, counts))} (Total Val: {len(val_cases)} frames)")

    # --- SAVE JSONS ---
    # 1. splits_final.json (For nnU-Net training)
    with open(PROCESSED_DIR / 'splits_final.json', 'w') as f:
        json.dump(splits, f, indent=4)

    # 2. dataset.json (Metadata)
    dataset_json = {
        "channel_names": {"0": "MRI"}, 
        "labels": {"background": 0, "RV": 1, "Myocardium": 2, "LV": 3}, 
        "numTraining": len(case_identifiers), 
        "file_ending": ".nii.gz"
    }
    with open(PROCESSED_DIR / 'dataset.json', 'w') as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\nSuccess! Data linked in {PROCESSED_DIR}")

if __name__ == "__main__":
    create_dataset()