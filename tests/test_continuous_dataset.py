"""
ISL Connect — Continuous Dataset Foundation Tests (Phase 8A).

Validates:
1. Directory structure for continuous dataset (videos_raw, sequences, metadata, cache).
2. Master manifest template schema and non-fabrication guarantee.
3. Vocabulary and sentence grammar metadata structure.
4. Canonical clip ID generation (ISL_<TYPE>_<LABEL>_<SIGNER>_<SESSION>_<SAMPLE>).
5. Validation of clip types, signer IDs, session IDs, and sample numbers.
6. Strictly subject-disjoint signer-to-split mapping (S01-S03 train, S04 val, S05 test).
7. Manifest row writing and serialization integrity.
"""

import csv
import json
from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.video_recorder import (
    ContinuousVideoRecorder,
    VideoClipMetadata,
    VALID_CLIP_TYPES,
    VALID_SIGNER_IDS,
    SIGNER_SPLIT_MAP,
    OPENCV_AVAILABLE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTINUOUS_DIR = PROJECT_ROOT / "dataset" / "continuous"


# ---------------------------------------------------------------------------
# 1. Directory Structure Tests
# ---------------------------------------------------------------------------

def test_continuous_directory_structure():
    """Verify that all required continuous dataset folders exist."""
    assert CONTINUOUS_DIR.exists()
    assert (CONTINUOUS_DIR / "videos_raw").exists()
    assert (CONTINUOUS_DIR / "sequences_processed" / "isolated_signs").exists()
    assert (CONTINUOUS_DIR / "sequences_processed" / "continuous_phrases").exists()
    assert (CONTINUOUS_DIR / "landmark_cache").exists()
    assert (CONTINUOUS_DIR / "metadata").exists()

    # Signer folders in videos_raw
    for signer in ["S01", "S02", "S03", "S04", "S05"]:
        signer_dir = CONTINUOUS_DIR / "videos_raw" / signer
        assert signer_dir.exists(), f"Signer raw video directory missing: {signer_dir}"


def test_readme_documentation_exists():
    """Verify dataset documentation README exists and outlines protocols."""
    readme_path = CONTINUOUS_DIR / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")
    assert "Continuous Sign Language Video Dataset" in content
    assert "Naming Convention" in content
    assert "Recording Protocol" in content
    assert "Signer-Independent" in content
    assert "Subject-Disjoint" in content
    assert "S01" in content and "S05" in content


# ---------------------------------------------------------------------------
# 2. Manifest Schema & Non-Fabrication Tests
# ---------------------------------------------------------------------------

def test_manifest_schema_and_empty_template():
    """Verify master manifest exists, has exact columns, and contains 0 fake recordings."""
    manifest_path = CONTINUOUS_DIR / "metadata" / "dataset_manifest.csv"
    assert manifest_path.exists(), "dataset_manifest.csv missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        assert header is not None, "Manifest file is empty"

        expected_columns = [
            "clip_id", "file_path", "clip_type", "gloss_sequence",
            "target_english", "signer_id", "session_id", "sample_id",
            "start_frame", "end_frame", "total_frames", "duration_sec",
            "fps", "resolution", "lighting", "background", "split", "notes",
        ]
        assert header == expected_columns, f"Header mismatch. Got: {header}"

        # Ensure no fabricated or fake video records exist
        data_rows = [row for row in reader if row and any(cell.strip() for cell in row)]
        assert len(data_rows) == 0, (
            f"Expected empty manifest template (zero fake recordings), but found {len(data_rows)} rows."
        )


# ---------------------------------------------------------------------------
# 3. Vocabulary and Sentence Grammar Metadata Tests
# ---------------------------------------------------------------------------

def test_vocabulary_metadata():
    """Verify vocabulary.json contains expected categories and well-formed signs."""
    vocab_path = CONTINUOUS_DIR / "metadata" / "vocabulary.json"
    assert vocab_path.exists(), "vocabulary.json missing"

    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "total_signs" in data
    assert "categories" in data
    assert "signs" in data
    assert len(data["signs"]) >= 20, f"Expected at least 20 vocabulary signs, found {len(data['signs'])}"

    expected_fields = {"id", "gloss", "category", "english_translation", "hands", "movement_description"}
    categories_found = set()
    for sign in data["signs"]:
        assert expected_fields.issubset(set(sign.keys())), f"Missing keys in sign: {sign}"
        categories_found.add(sign["category"])

    assert "Greetings" in categories_found
    assert "Questions" in categories_found
    assert "Emergencies" in categories_found


def test_sentence_grammar_metadata():
    """Verify sentence_grammar.json contains ISL to English phrase mappings."""
    grammar_path = CONTINUOUS_DIR / "metadata" / "sentence_grammar.json"
    assert grammar_path.exists(), "sentence_grammar.json missing"

    with open(grammar_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "mappings" in data
    mappings = data["mappings"]
    assert len(mappings) >= 5

    for m in mappings:
        assert "gloss_sequence" in m
        assert "english_translation" in m
        assert "intent" in m
        assert isinstance(m["gloss_sequence"], list)
        assert len(m["gloss_sequence"]) >= 1


# ---------------------------------------------------------------------------
# 4. Filename & Clip ID Generation Tests
# ---------------------------------------------------------------------------

def test_clip_id_generation_valid():
    """Verify canonical naming format ISL_<TYPE>_<LABEL>_<SIGNER>_<SESSION>_<SAMPLE>."""
    clip_id_iso = ContinuousVideoRecorder.generate_clip_id("ISO", "WATER", "S01", "SES01", 1)
    assert clip_id_iso == "ISL_ISO_WATER_S01_SES01_001"

    clip_id_phr = ContinuousVideoRecorder.generate_clip_id("PHR", "HOW-ARE-YOU", "S04", "SES02", 25)
    assert clip_id_phr == "ISL_PHR_HOW-ARE-YOU_S04_SES02_025"

    # Gloss with spaces should be hyphenated
    clip_id_spaced = ContinuousVideoRecorder.generate_clip_id("PHR", "THANK YOU", "S02", "SES01", 3)
    assert clip_id_spaced == "ISL_PHR_THANK-YOU_S02_SES01_003"


def test_clip_id_generation_invalid():
    """Verify strict validation of parameters during clip ID generation."""
    # Invalid clip type
    with pytest.raises(ValueError, match="Invalid clip_type"):
        ContinuousVideoRecorder.generate_clip_id("VIDEO", "WATER", "S01", "SES01", 1)

    # Invalid signer ID
    with pytest.raises(ValueError, match="Invalid signer_id"):
        ContinuousVideoRecorder.generate_clip_id("ISO", "WATER", "S99", "SES01", 1)

    # Invalid session ID format
    with pytest.raises(ValueError, match="Invalid session_id"):
        ContinuousVideoRecorder.generate_clip_id("ISO", "WATER", "S01", "SESSION1", 1)

    # Invalid sample ID
    with pytest.raises(ValueError, match="sample_id must be between 1 and 999"):
        ContinuousVideoRecorder.generate_clip_id("ISO", "WATER", "S01", "SES01", 0)

    with pytest.raises(ValueError, match="sample_id must be between 1 and 999"):
        ContinuousVideoRecorder.generate_clip_id("ISO", "WATER", "S01", "SES01", 1000)


# ---------------------------------------------------------------------------
# 5. Signer-Independent Split Assignment Tests
# ---------------------------------------------------------------------------

def test_signer_split_assignment():
    """Verify leak-proof subject-disjoint train/val/test assignment."""
    assert ContinuousVideoRecorder.get_split_for_signer("S01") == "train"
    assert ContinuousVideoRecorder.get_split_for_signer("S02") == "train"
    assert ContinuousVideoRecorder.get_split_for_signer("S03") == "train"
    assert ContinuousVideoRecorder.get_split_for_signer("S04") == "val"
    assert ContinuousVideoRecorder.get_split_for_signer("S05") == "test"

    with pytest.raises(ValueError, match="Unknown signer ID"):
        ContinuousVideoRecorder.get_split_for_signer("S99")

    # Verify disjointness
    train_signers = {s for s, sp in SIGNER_SPLIT_MAP.items() if sp == "train"}
    val_signers = {s for s, sp in SIGNER_SPLIT_MAP.items() if sp == "val"}
    test_signers = {s for s, sp in SIGNER_SPLIT_MAP.items() if sp == "test"}

    assert train_signers.isdisjoint(val_signers)
    assert train_signers.isdisjoint(test_signers)
    assert val_signers.isdisjoint(test_signers)


# ---------------------------------------------------------------------------
# 6. Manifest Append & Output Path Tests
# ---------------------------------------------------------------------------

def test_get_output_path(tmp_path):
    """Verify output path calculation under raw videos directory."""
    recorder = ContinuousVideoRecorder(output_root=tmp_path)
    out_path = recorder.get_output_path("ISL_ISO_WATER_S01_SES01_001", "S01")
    assert out_path == tmp_path / "S01" / "ISL_ISO_WATER_S01_SES01_001.mp4"
    assert (tmp_path / "S01").exists()


def test_append_manifest_entry(tmp_path):
    """Verify writing metadata rows to a manifest CSV without corrupting columns."""
    test_manifest = tmp_path / "test_manifest.csv"
    recorder = ContinuousVideoRecorder(manifest_path=test_manifest)

    entry = VideoClipMetadata(
        clip_id="ISL_ISO_HELLO_S01_SES01_001",
        file_path="dataset/continuous/videos_raw/S01/ISL_ISO_HELLO_S01_SES01_001.mp4",
        clip_type="ISO",
        gloss_sequence='["HELLO"]',
        target_english="Hello",
        signer_id="S01",
        session_id="SES01",
        sample_id=1,
        start_frame=0,
        end_frame=89,
        total_frames=90,
        duration_sec=3.0,
        fps=30.0,
        resolution="640x480",
        lighting="ambient",
        background="plain",
        split="train",
        notes="Clean recording",
    )

    recorder.append_manifest_entry(entry)
    assert test_manifest.exists()

    with open(test_manifest, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        assert row["clip_id"] == "ISL_ISO_HELLO_S01_SES01_001"
        assert row["signer_id"] == "S01"
        assert row["split"] == "train"
        assert row["fps"] == "30.0"
        assert row["total_frames"] == "90"


# ---------------------------------------------------------------------------
# 7. Camera Pipeline Integration & Error Handling Tests
# ---------------------------------------------------------------------------

def test_preview_camera_invalid_device():
    """Verify preview_camera handles nonexistent camera device without crashing."""
    recorder = ContinuousVideoRecorder()
    if OPENCV_AVAILABLE:
        with pytest.raises(RuntimeError, match="Could not open camera device"):
            recorder.preview_camera(camera_index=9999)


def test_record_clip_pipeline_with_mock(tmp_path, monkeypatch):
    """Verify record_clip execution, file path calculation, and manifest update with mock camera."""
    import numpy as np

    tmp_videos = tmp_path / "videos_raw"
    tmp_manifest = tmp_path / "metadata" / "dataset_manifest.csv"
    recorder = ContinuousVideoRecorder(output_root=tmp_videos, manifest_path=tmp_manifest)

    # Mock VideoCapture
    class MockVideoCapture:
        def __init__(self, idx, *args):
            self.idx = idx
            self.frames = 0
            self.max_frames = 10

        def isOpened(self):
            return True

        def get(self, prop):
            return 640 if prop == 3 else 480

        def read(self):
            if self.frames < self.max_frames:
                self.frames += 1
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            return False, None

        def release(self):
            pass

    # Mock VideoWriter
    class MockVideoWriter:
        def __init__(self, path, fourcc, fps, shape):
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch()

        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            pass

    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
    monkeypatch.setattr(cv2, "VideoWriter", MockVideoWriter)

    meta = recorder.record_clip(
        clip_type="ISO",
        gloss="THANK-YOU",
        signer_id="S02",
        session_id="SES01",
        sample_id=2,
        duration_seconds=0.3,
        fps=30.0,
        show_preview=False,
    )

    assert meta.clip_id == "ISL_ISO_THANK-YOU_S02_SES01_002"
    assert meta.signer_id == "S02"
    assert meta.split == "train"
    assert meta.total_frames == 9 or meta.total_frames == 10

    # Verify manifest was created and has entry
    assert tmp_manifest.exists()
    with open(tmp_manifest, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["clip_id"] == "ISL_ISO_THANK-YOU_S02_SES01_002"
        assert reader[0]["split"] == "train"


def test_record_clip_prevents_overwrite(tmp_path):
    """Verify record_clip refuses to overwrite an existing clip file."""
    recorder = ContinuousVideoRecorder(output_root=tmp_path)
    clip_id = recorder.generate_clip_id("ISO", "HELLO", "S01", "SES01", 1)
    out_file = recorder.get_output_path(clip_id, "S01")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.touch()  # Simulate existing clip file

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        recorder.record_clip(
            clip_type="ISO",
            gloss="HELLO",
            signer_id="S01",
            session_id="SES01",
            sample_id=1,
        )
