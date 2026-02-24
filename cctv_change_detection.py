"""
CCTV Change Detection System
Detects significant changes in video footage, captures frames before and after change,
and logs timestamps.

Compatible with Google Colab.
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
# Run this cell first in Google Colab
"""
!pip install opencv-python-headless numpy matplotlib scikit-learn Pillow tqdm
"""

# ============================================================
# CELL 2: Imports
# ============================================================
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display, Image as IPImage, HTML
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CELL 3: Configuration
# ============================================================
class Config:
    """
    Central configuration for the CCTV Change Detection System.
    Adjust these parameters based on your CCTV footage.
    """
    # --- Detection Sensitivity ---
    # Lower = more sensitive (detects smaller changes)
    # Higher = less sensitive (only large changes)
    CHANGE_THRESHOLD = 30           # Pixel intensity difference threshold (0-255)
    MIN_CHANGED_AREA_PERCENT = 2.0  # Minimum % of frame area that must change
    MOTION_BLUR_KERNEL = 5          # Blur to reduce noise (must be odd number)
    
    # --- Frame Sampling ---
    FRAME_SKIP = 2                  # Process every Nth frame (1 = every frame, 2 = every other)
    BUFFER_SECONDS = 2              # Seconds of context to capture before/after event
    
    # --- Event Cooldown ---
    # Minimum seconds between two separate change events (avoids duplicate detections)
    MIN_EVENT_GAP_SECONDS = 3.0
    
    # --- Output ---
    OUTPUT_DIR = "cctv_output"      # Where to save results
    SAVE_VIDEO_CLIPS = False        # Set True to save short video clips (requires more memory)
    
    # --- Display ---
    DISPLAY_LIVE = True             # Show frame-by-frame processing in Colab
    DISPLAY_EVERY_N = 30            # Show visualization every N frames

config = Config()

# ============================================================
# CELL 4: Core Change Detection Engine
# ============================================================
class FrameBuffer:
    """Circular buffer to store recent frames for 'before' snapshots."""
    def __init__(self, max_seconds: float, fps: float):
        self.max_frames = max(1, int(max_seconds * fps))
        self.buffer = []

    def add(self, frame, timestamp):
        self.buffer.append((frame.copy(), timestamp))
        if len(self.buffer) > self.max_frames:
            self.buffer.pop(0)

    def get_earliest(self):
        return self.buffer[0] if self.buffer else (None, None)

    def get_latest(self):
        return self.buffer[-1] if self.buffer else (None, None)


class ChangeDetector:
    """
    Core engine that compares consecutive frames and detects changes.
    Uses background subtraction + frame differencing for robustness.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # MOG2 background subtractor — good for CCTV with gradual lighting changes
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=True
        )
        self.prev_frame_gray = None
        self.frame_area = None

    def preprocess(self, frame):
        """Convert to grayscale + blur to reduce sensor noise."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, 
                                (self.cfg.MOTION_BLUR_KERNEL, self.cfg.MOTION_BLUR_KERNEL), 
                                0)

    def compute_change_score(self, frame):
        """
        Returns (score, diff_mask) where score is % of frame that changed.
        Combines background subtraction + frame differencing.
        """
        if self.frame_area is None:
            h, w = frame.shape[:2]
            self.frame_area = h * w

        gray = self.preprocess(frame)

        # --- Method 1: Background subtraction ---
        fg_mask = self.bg_subtractor.apply(frame)
        # Remove shadows (value=127), keep foreground (value=255)
        fg_mask = np.where(fg_mask == 255, 255, 0).astype(np.uint8)

        # --- Method 2: Frame differencing ---
        diff_mask = np.zeros_like(gray)
        if self.prev_frame_gray is not None:
            diff = cv2.absdiff(gray, self.prev_frame_gray)
            _, diff_mask = cv2.threshold(diff, self.cfg.CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)

        self.prev_frame_gray = gray

        # Combine both methods (union)
        combined_mask = cv2.bitwise_or(fg_mask, diff_mask)

        # Morphological cleanup — remove tiny speckles
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        changed_pixels = np.sum(combined_mask > 0)
        score = (changed_pixels / self.frame_area) * 100.0
        return score, combined_mask


class CCTVChangeDetectionSystem:
    """
    Full pipeline: reads video → detects changes → saves before/after frames → generates report.
    """
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.detector = ChangeDetector(self.cfg)
        self.events = []
        self.last_event_time = -999.0
        
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        print(f"✅ Output directory: {os.path.abspath(self.cfg.OUTPUT_DIR)}")

    # ----------------------------------------------------------
    def analyze_video(self, video_path: str, start_time_str: str = None):
        """
        Main method. Analyzes a video file and detects all change events.
        
        Args:
            video_path: Path to the video file.
            start_time_str: Real-world start time of the recording, e.g. "2024-01-15 08:00:00".
                            If None, uses relative timestamps (0:00:00, 0:00:05, ...).
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps

        # Parse start time
        if start_time_str:
            try:
                start_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                print(f"⚠️  Could not parse start_time_str='{start_time_str}'. Using relative timestamps.")
                start_dt = None
        else:
            start_dt = None

        print(f"\n{'='*60}")
        print(f"📹 Video: {os.path.basename(video_path)}")
        print(f"   Resolution : {width}x{height}")
        print(f"   FPS        : {fps:.2f}")
        print(f"   Duration   : {self._fmt_seconds(duration_sec)}")
        print(f"   Frames     : {total_frames}")
        print(f"{'='*60}\n")

        frame_buffer = FrameBuffer(self.cfg.BUFFER_SECONDS, fps)
        self.events = []
        self.detector.prev_frame_gray = None
        self.detector.frame_area = None
        
        # Warm up background subtractor with first 30 frames silently
        print("🔧 Warming up background model...")
        for _ in range(min(30, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            self.detector.bg_subtractor.apply(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("✅ Background model ready.\n")

        frame_idx = 0
        processed = 0
        scores = []
        pending_event = None   # Holds an event waiting for its 'after' frame

        pbar = tqdm(total=total_frames, desc="Analyzing", unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            pbar.update(1)

            # Skip frames for speed
            if frame_idx % self.cfg.FRAME_SKIP != 0:
                continue
            
            processed += 1
            timestamp_sec = frame_idx / fps
            wall_time = self._to_wall_time(timestamp_sec, start_dt)
            
            frame_buffer.add(frame, (timestamp_sec, wall_time))
            score, diff_mask = self.detector.compute_change_score(frame)
            scores.append((timestamp_sec, score))

            # ---- Collect 'after' frame for pending event ----
            if pending_event is not None:
                elapsed = timestamp_sec - pending_event["timestamp_sec"]
                if elapsed >= self.cfg.BUFFER_SECONDS:
                    pending_event["frame_after"] = frame.copy()
                    pending_event["diff_mask"] = diff_mask.copy()
                    self._save_event(pending_event)
                    self.events.append(pending_event)
                    print(f"\n  ✅ Event #{len(self.events)} saved at {pending_event['wall_time']}")
                    pending_event = None

            # ---- Detect new change event ----
            change_detected = (
                score >= self.cfg.MIN_CHANGED_AREA_PERCENT
                and (timestamp_sec - self.last_event_time) >= self.cfg.MIN_EVENT_GAP_SECONDS
            )

            if change_detected and pending_event is None:
                self.last_event_time = timestamp_sec
                before_frame, _ = frame_buffer.get_earliest()
                if before_frame is None:
                    before_frame = frame.copy()

                pending_event = {
                    "event_id": len(self.events) + 1,
                    "timestamp_sec": timestamp_sec,
                    "wall_time": wall_time,
                    "change_score": round(score, 2),
                    "frame_before": before_frame,
                    "frame_after": None,
                    "diff_mask": None,
                }
                print(f"\n  🔴 Change detected at {wall_time} "
                      f"(score={score:.1f}%) — capturing 'after' frame...")

            # ---- Optional live display ----
            if self.cfg.DISPLAY_LIVE and processed % self.cfg.DISPLAY_EVERY_N == 0:
                self._display_live(frame, diff_mask, score, wall_time)

        # Handle event still waiting for 'after' frame at end of video
        if pending_event is not None:
            last_frame = frame_buffer.get_latest()[0]
            pending_event["frame_after"] = last_frame if last_frame is not None else pending_event["frame_before"]
            pending_event["diff_mask"] = diff_mask
            self._save_event(pending_event)
            self.events.append(pending_event)

        pbar.close()
        cap.release()

        print(f"\n\n{'='*60}")
        print(f"🏁 Analysis complete!")
        print(f"   Frames processed : {processed}")
        print(f"   Change events    : {len(self.events)}")
        print(f"   Output saved to  : {os.path.abspath(self.cfg.OUTPUT_DIR)}")
        print(f"{'='*60}\n")

        self._generate_report(scores, os.path.basename(video_path))
        self._show_all_events()
        return self.events

    # ----------------------------------------------------------
    def _save_event(self, event: dict):
        """Save before/after frames and diff mask to disk."""
        eid = event["event_id"]
        ts  = event["timestamp_sec"]
        prefix = os.path.join(self.cfg.OUTPUT_DIR, f"event_{eid:03d}_t{int(ts)}s")

        cv2.imwrite(f"{prefix}_before.jpg", event["frame_before"])
        cv2.imwrite(f"{prefix}_after.jpg",  event["frame_after"])
        
        if event["diff_mask"] is not None:
            # Color the diff mask for visibility
            colored_diff = cv2.applyColorMap(event["diff_mask"], cv2.COLORMAP_HOT)
            cv2.imwrite(f"{prefix}_diff.jpg", colored_diff)

        # Save metadata JSON
        meta = {
            "event_id":     eid,
            "timestamp_sec": round(ts, 3),
            "wall_time":    event["wall_time"],
            "change_score": event["change_score"],
            "files": {
                "before": f"{prefix}_before.jpg",
                "after":  f"{prefix}_after.jpg",
                "diff":   f"{prefix}_diff.jpg",
            }
        }
        with open(f"{prefix}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ----------------------------------------------------------
    def _generate_report(self, scores: list, video_name: str):
        """Generate a timeline plot of change scores."""
        if not scores:
            return
        
        times  = [s[0] for s in scores]
        values = [s[1] for s in scores]
        event_times = [e["timestamp_sec"] for e in self.events]

        plt.figure(figsize=(14, 5))
        plt.plot(times, values, color='steelblue', linewidth=0.8, alpha=0.7, label="Change Score (%)")
        plt.axhline(self.cfg.MIN_CHANGED_AREA_PERCENT, color='orange', linestyle='--',
                    linewidth=1.5, label=f"Threshold ({self.cfg.MIN_CHANGED_AREA_PERCENT}%)")
        
        for i, et in enumerate(event_times):
            plt.axvline(et, color='red', linewidth=1.5, alpha=0.8,
                        label="Event" if i == 0 else "")
            plt.text(et, max(values) * 0.85, f"E{i+1}", color='red', fontsize=8,
                     ha='center', fontweight='bold')

        plt.xlabel("Time (seconds)")
        plt.ylabel("Changed Area (%)")
        plt.title(f"CCTV Change Detection Timeline — {video_name}")
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(self.cfg.OUTPUT_DIR, "change_timeline.png")
        plt.savefig(plot_path, dpi=120)
        plt.show()
        print(f"📊 Timeline saved: {plot_path}")

    # ----------------------------------------------------------
    def _show_all_events(self):
        """Display a visual summary of all detected events in Colab."""
        if not self.events:
            print("No events detected.")
            return
        
        print(f"\n{'='*60}")
        print(f"  DETECTED EVENTS SUMMARY ({len(self.events)} events)")
        print(f"{'='*60}")

        for ev in self.events:
            eid = ev["event_id"]
            print(f"\n📍 Event #{eid} | Time: {ev['wall_time']} | Score: {ev['change_score']}%")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"Event #{eid} — {ev['wall_time']} (Change: {ev['change_score']}%)",
                         fontsize=13, fontweight='bold')
            
            b = cv2.cvtColor(ev["frame_before"], cv2.COLOR_BGR2RGB)
            a = cv2.cvtColor(ev["frame_after"],  cv2.COLOR_BGR2RGB)
            
            axes[0].imshow(b); axes[0].set_title("BEFORE"); axes[0].axis('off')
            axes[1].imshow(a); axes[1].set_title("AFTER");  axes[1].axis('off')
            
            if ev["diff_mask"] is not None:
                d = cv2.applyColorMap(ev["diff_mask"], cv2.COLORMAP_HOT)
                d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
                axes[2].imshow(d); axes[2].set_title("DIFF MASK"); axes[2].axis('off')
            else:
                axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()

    # ----------------------------------------------------------
    def _display_live(self, frame, diff_mask, score, wall_time):
        """Show current frame during processing (Colab)."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[0].imshow(rgb); axes[0].set_title(f"Frame @ {wall_time}"); axes[0].axis('off')
        
        d = cv2.applyColorMap(diff_mask, cv2.COLORMAP_HOT)
        d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        axes[1].imshow(d); axes[1].set_title(f"Diff Mask | Score: {score:.1f}%"); axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------
    def export_results_json(self):
        """Export all event metadata to a single JSON file."""
        results = []
        for ev in self.events:
            results.append({
                "event_id":      ev["event_id"],
                "timestamp_sec": ev["timestamp_sec"],
                "wall_time":     ev["wall_time"],
                "change_score":  ev["change_score"],
            })
        out_path = os.path.join(self.cfg.OUTPUT_DIR, "all_events.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results exported: {out_path}")
        return out_path

    # ----------------------------------------------------------
    @staticmethod
    def _fmt_seconds(secs: float) -> str:
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _to_wall_time(ts_sec: float, start_dt) -> str:
        if start_dt:
            t = start_dt + timedelta(seconds=ts_sec)
            return t.strftime("%Y-%m-%d %H:%M:%S")
        else:
            h = int(ts_sec // 3600)
            m = int((ts_sec % 3600) // 60)
            s = int(ts_sec % 60)
            ms = int((ts_sec % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ============================================================
# CELL 5: Demo / Test with Synthetic Video
# ============================================================
def create_synthetic_cctv_video(output_path: str = "test_cctv.mp4",
                                 duration_sec: int = 30,
                                 fps: int = 25) -> str:
    """
    Creates a synthetic CCTV-like test video with simulated change events.
    Use this if you don't have a real video to test with.
    """
    width, height = 640, 480
    total_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Background: dark grey room
    bg = np.full((height, width, 3), 40, dtype=np.uint8)
    # Add some static texture (walls, floor line)
    cv2.rectangle(bg, (0, 350), (width, height), (30, 30, 30), -1)
    cv2.line(bg, (0, 350), (width, 350), (60, 60, 60), 2)
    cv2.rectangle(bg, (50, 100), (200, 300), (55, 55, 55), -1)  # box/cabinet

    # Define events: (start_frame, end_frame, description, object)
    events = [
        (fps*5,  fps*8,  "Person enters from left",  "person"),
        (fps*12, fps*16, "Object appears on shelf",   "object"),
        (fps*22, fps*26, "Person leaves right side",  "person2"),
    ]

    rng = np.random.default_rng(42)

    for fidx in range(total_frames):
        frame = bg.copy()
        # Add small noise
        noise = rng.integers(-5, 5, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        for (start, end, desc, obj) in events:
            if start <= fidx < end:
                progress = (fidx - start) / max(1, end - start)
                if obj == "person":
                    x = int(progress * (width - 100))
                    cv2.rectangle(frame, (x, 200), (x+60, 340), (160, 120, 90), -1)
                    cv2.circle(frame, (x+30, 185), 25, (200, 160, 130), -1)
                elif obj == "object":
                    cv2.rectangle(frame, (300, 250), (380, 310), (80, 120, 200), -1)
                elif obj == "person2":
                    x = int((1 - progress) * (width - 100))
                    cv2.rectangle(frame, (x, 200), (x+60, 340), (90, 160, 120), -1)
                    cv2.circle(frame, (x+30, 185), 25, (130, 200, 160), -1)

        # Timestamp overlay
        ts = f"CAM01 | {(fidx // fps // 60):02d}:{(fidx // fps % 60):02d}:{(fidx % fps):02d}"
        cv2.putText(frame, ts, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        out.write(frame)

    out.release()
    print(f"✅ Synthetic test video created: {output_path} ({duration_sec}s @ {fps}fps)")
    return output_path


# ============================================================
# CELL 6: Quick Run (Edit this section)
# ============================================================
def run_demo():
    """Run a quick demo with the synthetic video."""
    print("🎬 Generating synthetic CCTV test video...")
    video_path = create_synthetic_cctv_video("test_cctv.mp4", duration_sec=30, fps=25)

    # ---- Customize config ----
    cfg = Config()
    cfg.CHANGE_THRESHOLD = 25
    cfg.MIN_CHANGED_AREA_PERCENT = 1.5
    cfg.FRAME_SKIP = 1
    cfg.BUFFER_SECONDS = 1.5
    cfg.MIN_EVENT_GAP_SECONDS = 2.0
    cfg.DISPLAY_LIVE = False   # Set True to see frame-by-frame (slower)
    cfg.OUTPUT_DIR = "cctv_output"

    # ---- Run detection ----
    system = CCTVChangeDetectionSystem(cfg)
    events = system.analyze_video(
        video_path=video_path,
        start_time_str="2024-06-01 08:00:00"  # Real-world start time (optional)
    )
    system.export_results_json()

    print(f"\n✅ Done! Found {len(events)} change event(s).")
    print(f"📁 All files in: {os.path.abspath(cfg.OUTPUT_DIR)}/")
    return events


# ============================================================
# CELL 7: Run on YOUR video
# ============================================================
def run_on_your_video(video_path: str, 
                       start_time_str: str = None,
                       threshold: float = 30,
                       min_area_pct: float = 2.0,
                       buffer_sec: float = 2.0):
    """
    Run on a real CCTV video.
    
    Args:
        video_path       : Path to your .mp4 / .avi / .mkv file
        start_time_str   : "YYYY-MM-DD HH:MM:SS" — actual recording start time
        threshold        : Pixel diff threshold (10-50 recommended)
        min_area_pct     : Min % of frame that must change (1-10 recommended)
        buffer_sec       : Seconds before/after event to capture
    
    Example:
        events = run_on_your_video(
            video_path="/content/drive/MyDrive/cctv_footage.mp4",
            start_time_str="2024-06-01 08:30:00",
            threshold=25,
            min_area_pct=1.5,
            buffer_sec=2.0
        )
    """
    cfg = Config()
    cfg.CHANGE_THRESHOLD = threshold
    cfg.MIN_CHANGED_AREA_PERCENT = min_area_pct
    cfg.BUFFER_SECONDS = buffer_sec
    cfg.DISPLAY_LIVE = False
    cfg.OUTPUT_DIR = "cctv_output"

    system = CCTVChangeDetectionSystem(cfg)
    events = system.analyze_video(video_path, start_time_str=start_time_str)
    system.export_results_json()
    return events


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Run demo by default
    run_demo()
