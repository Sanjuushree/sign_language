"""
ISL Connect — Continuous Video Dataset Recorder & Manifest Manager (Phase 8A).

Utility for capturing, standardizing, and registering multi-signer Indian Sign
Language (ISL) video clips conforming to the Phase 7 recording protocol:
- Validates signer IDs, session IDs, and vocabulary gloss labels.
- Generates canonical clip identifiers: ISL_<TYPE>_<LABEL>_<SIGNER>_<SESSION>_<SAMPLE>.mp4
- Strictly enforces subject-disjoint train/val/test split assignment.
- Records video using OpenCV (cv2.VideoCapture & cv2.VideoWriter) upon explicit user trigger.
- Automatically appends clip metadata to dataset_manifest.csv.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.image_processor import (
    adaptive_light_correction,
    assess_illumination,
)

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONTINUOUS_DIR = _PROJECT_ROOT / "dataset" / "continuous"
DEFAULT_RAW_VIDEOS_DIR = DEFAULT_CONTINUOUS_DIR / "videos_raw"
DEFAULT_MANIFEST_PATH = DEFAULT_CONTINUOUS_DIR / "metadata" / "dataset_manifest.csv"
DEFAULT_VOCAB_PATH = DEFAULT_CONTINUOUS_DIR / "metadata" / "vocabulary.json"

VALID_CLIP_TYPES = {"ISO", "PHR"}
VALID_SIGNER_IDS = {"S01", "S02", "S03", "S04", "S05"}
VALID_SPLITS = {"train", "val", "test"}

# Signer-independent split assignment mapping (Strictly Subject-Disjoint)
SIGNER_SPLIT_MAP = {
    "S01": "train",
    "S02": "train",
    "S03": "train",
    "S04": "val",
    "S05": "test",
}


@dataclass
class VideoClipMetadata:
    """Metadata record for an individual continuous or isolated sign video clip."""
    clip_id: str
    file_path: str
    clip_type: str
    gloss_sequence: str
    target_english: str
    signer_id: str
    session_id: str
    sample_id: int
    start_frame: int
    end_frame: int
    total_frames: int
    duration_sec: float
    fps: float
    resolution: str
    lighting: str
    background: str
    split: str
    notes: str = ""


class ContinuousVideoRecorder:
    """Manages continuous ISL video recording, filename generation, and manifest logging."""

    def __init__(
        self,
        output_root: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        vocab_path: Optional[Union[str, Path]] = None,
    ):
        self.output_root = Path(output_root) if output_root else DEFAULT_RAW_VIDEOS_DIR
        self.manifest_path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
        self.vocab_path = Path(vocab_path) if vocab_path else DEFAULT_VOCAB_PATH

        self.vocabulary: List[Dict[str, Any]] = self._load_vocabulary()

    def _load_vocabulary(self) -> List[Dict[str, Any]]:
        """Load registered vocabulary signs from JSON if present."""
        if self.vocab_path.exists():
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("signs", [])
            except Exception:
                pass
        return []

    @staticmethod
    def get_split_for_signer(signer_id: str) -> str:
        """
        Return the leak-proof split for a given signer.
        S01..S03 -> train, S04 -> val, S05 -> test.
        """
        signer_clean = str(signer_id).strip().upper()
        if signer_clean not in SIGNER_SPLIT_MAP:
            raise ValueError(
                f"Unknown signer ID '{signer_id}'. Must be one of {sorted(list(VALID_SIGNER_IDS))}."
            )
        return SIGNER_SPLIT_MAP[signer_clean]

    @staticmethod
    def generate_clip_id(
        clip_type: str,
        gloss: str,
        signer_id: str,
        session_id: str,
        sample_id: int,
    ) -> str:
        """
        Generate standardized clip ID: ISL_<TYPE>_<LABEL>_<SIGNER>_<SESSION>_<SAMPLE>.

        Example:
        --------
        generate_clip_id("ISO", "WATER", "S01", "SES01", 1)
        -> "ISL_ISO_WATER_S01_SES01_001"
        """
        c_type = clip_type.strip().upper()
        if c_type not in VALID_CLIP_TYPES:
            raise ValueError(f"Invalid clip_type '{clip_type}'. Must be 'ISO' or 'PHR'.")

        s_id = signer_id.strip().upper()
        if s_id not in VALID_SIGNER_IDS:
            raise ValueError(f"Invalid signer_id '{signer_id}'. Must be in {sorted(list(VALID_SIGNER_IDS))}.")

        ses_id = session_id.strip().upper()
        if not (ses_id.startswith("SES") and len(ses_id) == 5 and ses_id[3:].isdigit()):
            raise ValueError(f"Invalid session_id '{session_id}'. Must be format 'SES01', 'SES02', etc.")

        if sample_id < 1 or sample_id > 999:
            raise ValueError(f"sample_id must be between 1 and 999, got {sample_id}.")

        clean_gloss = gloss.strip().upper().replace(" ", "-")
        return f"ISL_{c_type}_{clean_gloss}_{s_id}_{ses_id}_{sample_id:03d}"

    def get_output_path(self, clip_id: str, signer_id: str) -> Path:
        """Compute destination file path in videos_raw/<SIGNER_ID>/<clip_id>.mp4."""
        signer_dir = self.output_root / signer_id.strip().upper()
        signer_dir.mkdir(parents=True, exist_ok=True)
        return signer_dir / f"{clip_id}.mp4"

    def append_manifest_entry(self, metadata: VideoClipMetadata) -> None:
        """Append a validated video clip record to dataset_manifest.csv."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "clip_id", "file_path", "clip_type", "gloss_sequence",
            "target_english", "signer_id", "session_id", "sample_id",
            "start_frame", "end_frame", "total_frames", "duration_sec",
            "fps", "resolution", "lighting", "background", "split", "notes",
        ]

        import csv
        file_exists = self.manifest_path.exists() and self.manifest_path.stat().st_size > 0

        with open(self.manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(metadata))
    def preview_camera(
        self,
        camera_index: int = 0,
        window_name: str = "ISL Connect — Camera Preview",
    ) -> bool:
        """
        Open live camera preview window so the signer can position themselves.
        Returns True when user confirms (presses ENTER or SPACE), or False if cancelled (presses 'q' or ESC).
        """
        if not OPENCV_AVAILABLE:
            raise RuntimeError(
                "OpenCV is not available. Ensure opencv-python-headless or opencv-python is installed."
            )

        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera device at index {camera_index}.")

        confirmed = False
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                display = frame.copy()
                cv2.putText(
                    display,
                    "CAMERA PREVIEW - Press SPACE or ENTER to Confirm | Q to Cancel",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                info = assess_illumination(frame, is_bgr=True)
                if info.is_low_light:
                    cv2.putText(
                        display,
                        f"LOW LIGHT DETECTED (Luminance {info.mean_luminance:.0f}/255) - Auto-Correct Active",
                        (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2,
                    )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(30) & 0xFF
                if key in (13, 32):  # Enter or Space
                    confirmed = True
                    break
                elif key in (27, ord("q"), ord("Q")):  # Esc or Q
                    confirmed = False
                    break
        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        return confirmed

    def record_clip(
        self,
        clip_type: str,
        gloss: str,
        signer_id: str,
        session_id: str,
        sample_id: int,
        target_english: str = "",
        duration_seconds: float = 3.0,
        fps: float = 30.0,
        lighting: str = "ambient",
        background: str = "plain",
        camera_index: int = 0,
        notes: str = "",
        show_preview: bool = False,
        auto_light_correction: bool = True,
        save_enhanced: bool = False,
    ) -> VideoClipMetadata:
        """
        Record a single video clip from the webcam upon explicit user execution.

        Parameters:
        -----------
        clip_type : 'ISO' or 'PHR'
        gloss : Canonical sign gloss (e.g., 'WATER')
        signer_id : 'S01'..'S05'
        session_id : 'SES01'
        sample_id : 1..999
        duration_seconds : Recording length (default 3.0s)
        show_preview : If True, displays live recording feed with REC overlay
        auto_light_correction : If True, automatically detects and lifts low-light frames
        """
        if not OPENCV_AVAILABLE:
            raise RuntimeError(
                "OpenCV is not available in the current environment. "
                "Ensure opencv-python-headless or opencv-python is installed."
            )

        clip_id = self.generate_clip_id(clip_type, gloss, signer_id, session_id, sample_id)
        out_file = self.get_output_path(clip_id, signer_id)
        if out_file.exists():
            raise FileExistsError(
                f"Recording already exists: '{out_file.name}'. "
                "Refusing to overwrite existing dataset clip to preserve data integrity."
            )
        split = self.get_split_for_signer(signer_id)

        # Open camera using DirectShow on Windows, fallback to default
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera device at index {camera_index}.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        resolution_str = f"{width}x{height}"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not initialize VideoWriter for destination {out_file}.")

        frames_to_record = int(duration_seconds * fps)
        recorded_frames = 0
        start_time = time.time()
        window_title = "ISL Connect — Recording"
        low_light_detected = False

        try:
            while recorded_frames < frames_to_record:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if auto_light_correction or save_enhanced:
                    enhanced_frame, was_corrected, _ = adaptive_light_correction(frame, is_bgr=True)
                    if was_corrected:
                        low_light_detected = True
                else:
                    enhanced_frame = frame

                # Determine whether raw or enhanced frame is committed to disk
                frame_to_write = enhanced_frame if save_enhanced else frame

                writer.write(frame_to_write)
                recorded_frames += 1

                if show_preview:
                    try:
                        rec_display = enhanced_frame.copy() if low_light_detected else frame.copy()
                        cv2.circle(rec_display, (25, 25), 8, (0, 0, 255), -1)
                        status_txt = f"REC {recorded_frames}/{frames_to_record} ({gloss})"
                        if low_light_detected:
                            status_txt += " [ENHANCED]" if save_enhanced else " [RAW LOW-LIGHT]"
                        cv2.putText(
                            rec_display,
                            status_txt,
                            (42, 31),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imshow(window_title, rec_display)
                        cv2.waitKey(1)
                    except Exception:
                        pass
        finally:
            writer.release()
            cap.release()
            if show_preview:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

        actual_duration = round(time.time() - start_time, 2)
        try:
            rel_path = str(out_file.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            rel_path = str(out_file).replace("\\", "/")

        storage_tag = "enhanced_video" if save_enhanced else "raw_video"
        lighting_tag = "low_light_detected" if low_light_detected else "normal_lighting"
        final_notes = f"{notes}; {storage_tag}; {lighting_tag}" if notes else f"{storage_tag}; {lighting_tag}"

        metadata = VideoClipMetadata(
            clip_id=clip_id,
            file_path=rel_path,
            clip_type=clip_type,
            gloss_sequence=f'["{gloss}"]',
            target_english=target_english or gloss.title(),
            signer_id=signer_id,
            session_id=session_id,
            sample_id=sample_id,
            start_frame=0,
            end_frame=recorded_frames - 1 if recorded_frames > 0 else 0,
            total_frames=recorded_frames,
            duration_sec=actual_duration,
            fps=fps,
            resolution=resolution_str,
            lighting=lighting,
            background=background,
            split=split,
            notes=final_notes,
        )

        self.append_manifest_entry(metadata)
        return metadata


def interactive_cli() -> None:
    """Interactive command-line interface for manual video recording."""
    print("=" * 65)
    print("ISL Connect — Continuous Dataset Video Recorder (Phase 8A)")
    print("=" * 65)

    recorder = ContinuousVideoRecorder()
    vocab = recorder.vocabulary

    if not vocab:
        print("⚠️ Warning: No vocabulary found in vocabulary.json.")
    else:
        print(f"Loaded {len(vocab)} registered vocabulary signs.")

    print("\nSigner IDs: S01, S02, S03 (Train) | S04 (Val) | S05 (Held-Out Test)")
    signer_id = input("Enter Signer ID (e.g. S01): ").strip().upper()
    if signer_id not in VALID_SIGNER_IDS:
        print(f"Error: Invalid signer ID '{signer_id}'. Aborting.")
        return

    session_id = input("Enter Session ID (e.g. SES01): ").strip().upper()
    clip_type = input("Enter Clip Type (ISO for word, PHR for phrase) [ISO]: ").strip().upper() or "ISO"
    gloss = input("Enter Gloss Label (e.g. WATER, HELLO): ").strip().upper()
    sample_id_str = input("Enter Sample Number (e.g. 1): ").strip()
    sample_id = int(sample_id_str) if sample_id_str.isdigit() else 1
    duration_str = input("Enter Recording Duration in Seconds [3.0]: ").strip()
    duration = float(duration_str) if duration_str else 3.0

    print(f"\nTarget Clip: {recorder.generate_clip_id(clip_type, gloss, signer_id, session_id, sample_id)}")
    print(f"Assigned Split: {recorder.get_split_for_signer(signer_id)}")

    save_mode = input("Save format: [R]aw camera footage (default) or [E]nhanced? [R]: ").strip().upper()
    save_enhanced = (save_mode == "E")

    preview_choice = input("\nPreview camera before recording? (y/n) [y]: ").strip().lower()
    if preview_choice != "n":
        print("Opening camera preview window...\nPress SPACE or ENTER in the preview window to start recording (or Q to cancel).")
        try:
            confirmed = recorder.preview_camera()
            if not confirmed:
                print("Recording cancelled during preview.")
                return
        except Exception as e:
            print(f"Could not open preview window: {e}. Falling back to prompt.")
            confirm = input("Press ENTER to start recording (or 'q' to cancel): ").strip()
            if confirm.lower() == "q":
                print("Cancelled.")
                return
    else:
        confirm = input("Press ENTER to start recording (or 'q' to cancel): ").strip()
        if confirm.lower() == "q":
            print("Cancelled.")
            return

    print("\n🔴 Recording in progress... Please perform the sign gesture.")
    try:
        meta = recorder.record_clip(
            clip_type=clip_type,
            gloss=gloss,
            signer_id=signer_id,
            session_id=session_id,
            sample_id=sample_id,
            duration_seconds=duration,
            show_preview=True,
            save_enhanced=save_enhanced,
        )
        print(f"\n✅ Recording saved successfully:")
        print(f"   File: {meta.file_path}")
        print(f"   Frames: {meta.total_frames} ({meta.duration_sec}s @ {meta.fps} fps)")
        print(f"   Split: {meta.split}")
        print(f"   Notes: {meta.notes}")
        print(f"   Logged in: {recorder.manifest_path}")
    except Exception as e:
        print(f"❌ Recording error: {e}")


def cli_main() -> None:
    """Entry point supporting both CLI argument flags and interactive mode."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ISL Connect — Continuous Dataset Video Recorder (Phase 8B)",
        add_help=True,
    )
    parser.add_argument("--signer", type=str, help="Signer ID (S01..S05)")
    parser.add_argument("--session", type=str, help="Session ID (e.g. SES01)")
    parser.add_argument("--type", type=str, default="ISO", help="Clip type: ISO or PHR (default: ISO)")
    parser.add_argument("--gloss", type=str, help="Sign gloss label (e.g. HELLO, WATER)")
    parser.add_argument("--sample", type=int, default=1, help="Sample number 1..999 (default: 1)")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds (default: 3.0)")
    parser.add_argument("--save-enhanced", action="store_true", help="Save enhanced video instead of raw")
    parser.add_argument("--no-preview", action="store_true", help="Skip live camera preview")

    args = parser.parse_args()

    # If any core argument was passed, run non-interactive / direct recording
    if args.signer or args.session or args.gloss:
        if not (args.signer and args.session and args.gloss):
            print("Error: --signer, --session, and --gloss are all required when running in flag mode.")
            return

        recorder = ContinuousVideoRecorder()
        signer_id = args.signer.strip().upper()
        session_id = args.session.strip().upper()
        clip_type = args.type.strip().upper()
        gloss = args.gloss.strip().upper()
        sample_id = args.sample
        duration = args.duration

        clip_id = recorder.generate_clip_id(clip_type, gloss, signer_id, session_id, sample_id)
        split = recorder.get_split_for_signer(signer_id)
        print("=" * 65)
        print("ISL Connect — Direct Video Recording")
        print("=" * 65)
        print(f"Target Clip:    {clip_id}")
        print(f"Assigned Split: {split}")
        print(f"Duration:       {duration}s")
        print(f"Save Mode:      {'Enhanced' if args.save_enhanced else 'Raw Camera (Default)'}")

        if not args.no_preview:
            print("\nOpening camera preview window...\nPress SPACE or ENTER in the preview window to start recording (or Q to cancel).")
            try:
                confirmed = recorder.preview_camera()
                if not confirmed:
                    print("Recording cancelled during preview.")
                    return
            except Exception as e:
                print(f"Preview unavailable: {e}. Press ENTER to proceed with recording.")
                input()

        print("\n🔴 Recording in progress... Please perform the sign gesture.")
        try:
            meta = recorder.record_clip(
                clip_type=clip_type,
                gloss=gloss,
                signer_id=signer_id,
                session_id=session_id,
                sample_id=sample_id,
                duration_seconds=duration,
                show_preview=True,
                save_enhanced=args.save_enhanced,
            )
            print(f"\n✅ Recording saved successfully:")
            print(f"   File: {meta.file_path}")
            print(f"   Frames: {meta.total_frames} ({meta.duration_sec}s @ {meta.fps} fps)")
            print(f"   Split: {meta.split}")
            print(f"   Notes: {meta.notes}")
            print(f"   Logged in: {recorder.manifest_path}")
        except Exception as e:
            print(f"❌ Recording error: {e}")
    else:
        interactive_cli()


if __name__ == "__main__":
    cli_main()
