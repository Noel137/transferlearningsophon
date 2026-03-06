import os
import sys
import csv
import math
import glob
import argparse
import torch
import uproot
import numpy as np
from tqdm import tqdm
from math import cos, sin, sinh
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from networks.example_ParticleTransformer_sophon import get_model

TARGET_EVENTS_PER_CLASS = 100_000
MAX_PART = 128
STEP_SIZE = 5000
TREE_NAME = "tree"
ROOT_DIR = "/data/JetClass/Pythia/train_100M"
OUTPUT_DIR = "/data/embeddings_train"
SKIP_IF_EXISTS = False

# ---------------------------------------------------------------------------
# The 17 derived Sophon input features (matches pretrained model's input_dim)
# ---------------------------------------------------------------------------
SOPHON_FEATURE_NAMES = [
    "part_pt_scale_log",
    "part_e_scale_log",
    "part_logptrel",
    "part_logerel",
    "part_deltaR",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "part_d0",
    "part_d0err",
    "part_dz",
    "part_dzerr",
    "part_deta",
    "part_dphi",
]
NUM_SOPHON_FEATURES = len(SOPHON_FEATURE_NAMES)  # 17


class_prefixes = [
    "HToBB", "HToCC", "HToGG", "HToWW4Q", "HToWW2Q1L", 
    "ZToQQ", "WToQQ", "TTBar", "TTBarLep", "ZJetsToNuNu"
]

JET_CLASSES = {}
for prefix in class_prefixes:
    # Look inside the training folder for all files starting with the particle name
    search_pattern = os.path.join(ROOT_DIR, f"{prefix}_*.root")
    matched_files = [os.path.basename(f) for f in glob.glob(search_pattern)]
    matched_files.sort() # Keeps them in a neat order (000, 001, 002...)
    
    # Fix the naming quirk for ZJetsToNuNu to match your previous code
    out_name = "ZToNuNu" if prefix == "ZJetsToNuNu" else prefix
    
    JET_CLASSES[prefix] = {
        "files": matched_files,
        "output": f"{out_name}_train_inference_with_embedding_nfr.csv"
    }

# Quick safety check to ensure we actually found the files
for k, v in JET_CLASSES.items():
    if len(v["files"]) == 0:
        print(f"WARNING: No files found for {k} in {ROOT_DIR}")



particle_keys = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_deta", "part_dphi", "part_d0val", "part_d0err",
    "part_dzval", "part_dzerr", "part_charge",
    "part_isChargedHadron", "part_isNeutralHadron",
    "part_isPhoton", "part_isElectron", "part_isMuon",
]

scalar_keys_for_model = [
    "jet_pt", "jet_eta", "jet_phi",
    "jet_energy", "jet_nparticles", "jet_sdmass",
    "jet_tau1", "jet_tau2", "jet_tau3", "jet_tau4",
]

label_keys = [
    "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg",
    "label_H4q", "label_Hqql", "label_Zqq", "label_Wqq",
    "label_Tbqq", "label_Tbl",
]

pf_keys = particle_keys + label_keys + scalar_keys_for_model

label_names = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]

class DummyDataConfig:
    input_dicts = {"pf_features": list(range(NUM_SOPHON_FEATURES))}
    input_names = ["pf_points"]
    input_shapes = {"pf_points": (MAX_PART, NUM_SOPHON_FEATURES)}
    label_names = ["label"]
    num_classes = 10

def _norm(x, subtract, multiply, clip_lo=-5.0, clip_hi=5.0):
    """Apply (x - subtract) * multiply then clip to [clip_lo, clip_hi]."""
    return np.clip((x - subtract) * multiply, clip_lo, clip_hi)


def compute_sophon_features(arrays, i, keep_idx=None):
    """Derive the 17 Sophon input features from raw particle arrays.

    Preprocessing matches the official JetClassII_full.yaml config exactly:
      - momentum/energy logs are computed from jet_pt*500-scaled values
      - 5 kinematic features are shift/scale normalised then clipped to [-5,5]
      - d0 / dz use tanh transform;  d0err / dzerr are clipped to [0,1]

    Returns an (n_part, 17) float32 array.
    """
    get = (lambda k: arrays[k][i][keep_idx]) if keep_idx is not None else (lambda k: arrays[k][i])

    px = get("part_px")
    py = get("part_py")
    energy = get("part_energy")

    jet_pt_val = float(arrays["jet_pt"][i])
    jet_energy_val = float(arrays["jet_energy"][i])

    eps = 1e-20
    jet_pt_safe = max(jet_pt_val, eps)
    jet_energy_safe = max(jet_energy_val, eps)

    # Scaled kinematics (official: part_*_scale = part_* * 500 / jet_pt)
    pt = np.sqrt(px ** 2 + py ** 2)
    pt_scale = pt * 500.0 / jet_pt_safe
    energy_scale = energy * 500.0 / jet_pt_safe

    # Logarithmic features with normalization (subtract, multiply, clip)
    pt_scale_log = _norm(np.log(np.clip(pt_scale, eps, None)),
                         subtract=1.7, multiply=0.7)
    e_scale_log  = _norm(np.log(np.clip(energy_scale, eps, None)),
                         subtract=2.0, multiply=0.7)
    logptrel     = _norm(np.log(np.clip(pt / jet_pt_safe, eps, None)),
                         subtract=-4.7, multiply=0.7)
    logerel      = _norm(np.log(np.clip(energy / jet_energy_safe, eps, None)),
                         subtract=-4.7, multiply=0.7)

    deta = get("part_deta")
    dphi = get("part_dphi")
    deltaR = _norm(np.sqrt(deta ** 2 + dphi ** 2),
                   subtract=0.2, multiply=4.0)

    # Impact parameters: tanh transform for d0/dz, clip [0,1] for errors
    d0    = np.tanh(get("part_d0val"))
    d0err = np.clip(get("part_d0err"), 0.0, 1.0)
    dz    = np.tanh(get("part_dzval"))
    dzerr = np.clip(get("part_dzerr"), 0.0, 1.0)

    feats = np.stack([
        pt_scale_log,
        e_scale_log,
        logptrel,
        logerel,
        deltaR,
        get("part_charge"),
        get("part_isChargedHadron"),
        get("part_isNeutralHadron"),
        get("part_isPhoton"),
        get("part_isElectron"),
        get("part_isMuon"),
        d0,
        d0err,
        dz,
        dzerr,
        deta,
        dphi,
    ], axis=1).astype(np.float32)

    return feats


def build_pf_arrays(arrays, i):
    """Build raw NumPy arrays for high-speed batching on the CPU."""
    n_part = arrays["part_px"][i].shape[0]

    if n_part > MAX_PART:
        px = arrays["part_px"][i]
        py = arrays["part_py"][i]
        pt = np.sqrt(px * px + py * py)
        keep_idx = np.argsort(pt)[::-1][:MAX_PART]
        n_part = MAX_PART
    else:
        keep_idx = None

    get = (lambda k: arrays[k][i][keep_idx]) if keep_idx is not None else (lambda k: arrays[k][i])

    jet_pt_val = float(arrays["jet_pt"][i])
    jet_pt_safe = max(jet_pt_val, 1e-20)
    lv = np.stack([
        get("part_px") * 500.0 / jet_pt_safe,
        get("part_py") * 500.0 / jet_pt_safe,
        get("part_pz") * 500.0 / jet_pt_safe,
        get("part_energy") * 500.0 / jet_pt_safe,
    ], axis=1).astype(np.float32)

    sophon_feats = compute_sophon_features(arrays, i, keep_idx)

    lv_padded = np.zeros((MAX_PART, 4), dtype=np.float32)
    lv_padded[:n_part] = lv

    feat_padded = np.zeros((MAX_PART, NUM_SOPHON_FEATURES), dtype=np.float32)
    feat_padded[:n_part] = sophon_feats

    # Notice: Returning numpy arrays, NOT torch.tensors
    return feat_padded, lv_padded, (lv_padded[:, 3] != 0)

def get_truth_label(arrays, i):
    labs = np.array([arrays[k][i] for k in label_keys])
    y = int(np.argmax(labs))
    return y, label_names[y]

def jet_masses(arrays, i):
    jet_sdmass = float(arrays["jet_sdmass"][i])
    pt = float(arrays["jet_pt"][i])
    eta = float(arrays["jet_eta"][i])
    phi = float(arrays["jet_phi"][i])
    E = float(arrays["jet_energy"][i])
    px = pt * cos(phi)
    py = pt * sin(phi)
    pz = pt * sinh(eta)
    m2 = max(E * E - (px * px + py * py + pz * pz), 0.0)
    return jet_sdmass, math.sqrt(m2), pt, eta, phi

def process_class(class_name, class_info, model, device, num_classes=10):
    output_path = os.path.join(OUTPUT_DIR, class_info["output"])

    if SKIP_IF_EXISTS and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        print(f"Skipping {class_name} (exists): {output_path}")
        return 0

    print(f"\nProcessing {class_name}")

    root_files = class_info["files"]
    total_written = 0
    wrote_header = False
    target = TARGET_EVENTS_PER_CLASS

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        paths = [f"{os.path.join(ROOT_DIR, fn)}:{TREE_NAME}" for fn in root_files]

        it = uproot.iterate(
            paths,
            expressions=pf_keys,
            step_size=STEP_SIZE,
            library="np",
            report=True,
        )

        for batch_idx, (arrays, report) in enumerate(it):
            batch_len = len(arrays["jet_pt"])

            source_file = os.path.basename(getattr(report, "file_path", "unknown"))
            batch_start_entry = getattr(report, "entry_start", 0)

            # --- HIGH-SPEED BATCHING LOGIC ---
            batch_features, batch_lvs, batch_masks, metadata_rows = [], [], [], []

            # 1. Quickly format all events in this chunk on the CPU
            for i in range(batch_len):
                if target is not None and (total_written + len(batch_features)) >= target:
                    break
                
                feat_padded, lv_padded, mask = build_pf_arrays(arrays, i)
                batch_features.append(feat_padded)
                batch_lvs.append(lv_padded)
                batch_masks.append(mask)

                truth_label, label_name = get_truth_label(arrays, i)
                jet_sdmass, jet_mass, pt, eta, phi = jet_masses(arrays, i)
                metadata_rows.append([truth_label, label_name, jet_sdmass, jet_mass, pt, eta, phi])

            if not batch_features:
                break

            # 2. Convert the entire chunk to tensors and send to GPU ONCE
            feat_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32).transpose(1, 2).to(device)
            lv_tensor = torch.tensor(np.array(batch_lvs), dtype=torch.float32).transpose(1, 2).to(device)
            mask_tensor = torch.tensor(np.array(batch_masks), dtype=torch.bool).unsqueeze(1).to(device)

            # 3. Process all events simultaneously
            with torch.no_grad():
                out = model(None, feat_tensor, lv_tensor, mask_tensor)

            if isinstance(out, tuple):
                logits, embeddings = out
                logits_np = logits.detach().cpu().numpy()
            else:
                embeddings = out
                logits_np = np.zeros((len(batch_features), num_classes), dtype=np.float32)

            embeddings_np = embeddings.detach().cpu().numpy()

            # 4. Write the results to the CSV
            for idx in range(len(metadata_rows)):
                if not wrote_header:
                    base = ["source_file", "entry_index", "row_index", "truth_label", "label_name", "jet_sdmass", "jet_mass", "jet_pt", "jet_eta", "jet_phi"]
                    logit_cols = [f"logit_{j}" for j in range(logits_np.shape[1])]
                    emb_cols = [f"emb_{j}" for j in range(embeddings_np.shape[-1])]
                    writer.writerow(base + logit_cols + emb_cols)
                    wrote_header = True

                row = [
                    source_file, int(batch_start_entry + idx), total_written,
                    *metadata_rows[idx], *logits_np[idx].tolist(), *embeddings_np[idx].tolist()
                ]
                writer.writerow(row)
                total_written += 1

            print(f"  -> Processed {total_written}/{target if target else 'ALL'} events...")

            if target is not None and total_written >= target:
                break

    print(f"{class_name}: saved {total_written:,} rows -> {output_path}")
    return total_written

def main():
    parser = argparse.ArgumentParser(description="JetClass Sophon inference & embedding extraction")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained model.pt weights file. "
                             "If omitted, runs with random-init weights (baseline).")
    args = parser.parse_args()

    data_config = DummyDataConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint (if any) and detect num_classes ---
    num_classes = 10  # default for random-init
    checkpoint_state = None

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            sys.exit(f"Checkpoint not found: {ckpt_path}")
        raw = torch.load(str(ckpt_path), map_location=device)
        checkpoint_state = raw["model_state_dict"] if "model_state_dict" in raw else raw
        # Detect num_classes from FC layer weight shape
        for key in ["mod.fc.0.weight", "fc.0.weight"]:
            if key in checkpoint_state:
                num_classes = checkpoint_state[key].shape[0]
                print(f"Detected num_classes={num_classes} from checkpoint")
                break

    # --- Create model with correct num_classes ---
    print(f"Loading model on {device} (num_classes={num_classes})...")
    model, _ = get_model(data_config, num_classes=num_classes, export_embed=True)

    if checkpoint_state is not None:
        # Try loading; if keys lack 'mod.' prefix, remap them
        missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
        if len(unexpected) > 0 and all(not k.startswith("mod.") for k in checkpoint_state):
            remapped = {"mod." + k: v for k, v in checkpoint_state.items()}
            missing, unexpected = model.load_state_dict(remapped, strict=False)
            print(f"Remapped checkpoint keys (added 'mod.' prefix)")
        loaded = len(checkpoint_state) - len(unexpected)
        print(f"Loaded pretrained weights from {ckpt_path} "
              f"({loaded} params loaded, {len(missing)} missing, {len(unexpected)} unexpected)")
    else:
        print("No --checkpoint provided; using random-init weights")

    model.eval().to(device)
    print("Model ready")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("JetClass embedding generator")
    print(f"classes: {len(JET_CLASSES)}")
    print(f"events per class: {TARGET_EVENTS_PER_CLASS if TARGET_EVENTS_PER_CLASS else 'ALL'}")
    print(f"root dir: {ROOT_DIR}")
    print(f"out dir: {OUTPUT_DIR}")
    print(f"skip existing: {SKIP_IF_EXISTS}")

    total_events = 0
    for class_name, class_info in JET_CLASSES.items():
        total_events += process_class(class_name, class_info, model, device, num_classes)

    print(f"Done. Total events written: {total_events:,}")
    print(f"Outputs in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
