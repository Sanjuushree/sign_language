# ISL Connect — Continuous Sign Language Video Dataset

This directory contains the foundational specifications, metadata schemas, and storage structure for the **Indian Sign Language (ISL) Continuous Recognition & Sentence Translation Dataset**.

---

## 1. Directory Structure

```text
dataset/continuous/
├── README.md                      # This documentation file
├── metadata/
│   ├── vocabulary.json            # 25 core ISL conversational signs with metadata
│   ├── sentence_grammar.json      # Mapping rules from gloss sequences to English
│   └── dataset_manifest.csv       # Master CSV registry of all recorded clips
├── videos_raw/                    # Uncompressed camera video captures (.mp4)
│   ├── S01/                       # Signer 01 (Training Split)
│   ├── S02/                       # Signer 02 (Training Split)
│   ├── S03/                       # Signer 03 (Training Split)
│   ├── S04/                       # Signer 04 (Validation Split)
│   └── S05/                       # Signer 05 (Held-Out Test Split)
├── sequences_processed/           # Normalized, cropped, CFR-standardized clips
│   ├── isolated_signs/            # Single sign gloss clips
│   └── continuous_phrases/        # Multi-sign sentence clips
└── landmark_cache/                # Extracted spatio-temporal keypoint arrays (.npz)
```

---

## 2. Standardized Video Naming Convention

Every video recording must follow a strict, deterministic naming pattern:

$$\texttt{ISL\_<TYPE>\_<LABEL>\_<SIGNER\_ID>\_<SESSION\_ID>\_<SAMPLE\_NUM>.mp4}$$

### Field Definitions:
- **`TYPE`**: `ISO` for isolated single-word signs; `PHR` for multi-sign continuous phrases.
- **`LABEL`**: Standard gloss name in uppercase using hyphens (e.g., `WATER`, `THANK-YOU`, `HOW-ARE-YOU`).
- **`SIGNER_ID`**: Three-character signer identifier: `S01`, `S02`, `S03`, `S04`, `S05`.
- **`SESSION_ID`**: Five-character session identifier: `SES01`, `SES02`, `SES03`.
- **`SAMPLE_NUM`**: Three-digit zero-padded repetition number: `001`, `002`, `015`.

### Canonical Examples:
- `ISL_ISO_WATER_S01_SES01_001.mp4` — Signer 1, Session 1, repetition 1 of isolated sign "WATER".
- `ISL_ISO_HELP_S02_SES02_012.mp4` — Signer 2, Session 2, repetition 12 of isolated sign "HELP".
- `ISL_PHR_YOU-NAME-WHAT_S04_SES01_003.mp4` — Signer 4, Session 1, repetition 3 of phrase "What is your name?".

---

## 3. Recording Protocol & Setup

To ensure machine learning models generalize to novel users and environments:

1. **Framing (Medium Shot):**
   - The signer must be captured from the waist up to approximately 20 cm above the head.
   - Arms and hands must remain within the frame at all times during full extension, with a 15% margin on both sides.
2. **Camera Specifications:**
   - Resolution: 1080p (1920×1080) or 720p (1280×720).
   - Frame Rate: Constant **30 frames per second (CFR)**. Do not use variable frame rates.
3. **Lighting Conditions (3 Environments):**
   - `studio`: Frontal soft lighting without harsh facial shadows.
   - `daylight`: Diffuse natural daylight from a side window.
   - `ambient`: Standard indoor overhead warm or white room lighting.
4. **Backgrounds (3 Setups):**
   - `plain`: Solid matte light-grey or off-white wall.
   - `domestic`: Living room, home bookshelf, or curtain background.
   - `office`: Classroom, office desk, or doorway background.
5. **Signing Execution:**
   - Begin with hands relaxed at resting position for 0.5s.
   - Execute the sign clearly at conversational speed.
   - Return hands to rest for 0.5s before stopping recording.

---

## 4. Signer-Independent (Subject-Disjoint) Splitting

To prevent data leakage, recordings from the same individual must **never appear across both training and test partitions**:

| Split | Assigned Signers | Allocation | Role in Pipeline |
|---|---|:---:|---|
| **Training (`train`)** | **S01, S02, S03** | **60%** | Feature extraction, sequence model training, gradient updates. |
| **Validation (`val`)** | **S04** | **20%** | Hyperparameter tuning, early stopping, model architecture selection. |
| **Testing (`test`)** | **S05** (Strictly Held-Out) | **20%** | Final benchmark evaluation on an entirely unfamiliar signer. |

> [!CAUTION]
> **Data Leakage Rule:** No recordings of Signer `S05` may ever be used during training, feature scaling, or threshold tuning.

---

## 5. Master Manifest Format (`metadata/dataset_manifest.csv`)

Every recorded video must be logged in `dataset_manifest.csv` with the following columns:

```csv
clip_id,file_path,clip_type,gloss_sequence,target_english,signer_id,session_id,sample_id,start_frame,end_frame,total_frames,duration_sec,fps,resolution,lighting,background,split,notes
```

- `clip_id`: Matches the filename without the `.mp4` extension.
- `file_path`: Relative path from the repository root.
- `start_frame` & `end_frame`: Frame indices where active motion begins and ends (annotated via MediaPipe motion tracking or review).
- `split`: Strictly locked to `train`, `val`, or `test` according to the signer assignment.
