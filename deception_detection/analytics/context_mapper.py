import json
import bisect
import numpy as np
from pathlib import Path
import logging

class ContextMapper:
    def __init__(self, manifest_path: str = None):
        """
        O(log N) Binary Interval Search Engine for mapping chronological events
        from an investigative manifest to the 30 FPS feature tensor.
        """
        self.logger = logging.getLogger("Context_Mapper")
        self.intervals = []  # Tuples: (start_ms, end_ms, phase_label, question_id)
        self.starts = []     # Pre-extracted for fast bisect

        if manifest_path is None:
            self.logger.info("No session_manifest provided. Contextual schema will default to safe NaNs.")
            return

        path = Path(manifest_path)
        if not path.exists():
            self.logger.warning(f"session_manifest not found at {path}. Defaulting to safe NaNs.")
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Parse intervals
            for item in data:
                self.intervals.append((
                    float(item.get("start_ms", 0.0)),
                    float(item.get("end_ms", 0.0)),
                    str(item.get("phase_label", "N/A")),
                    int(item.get("question_id", -1))
                ))
            
            # Sort by start time guarantees O(log N) bisect safety
            self.intervals.sort(key=lambda x: x[0])
            self.starts = [x[0] for x in self.intervals]
            
            self.logger.info(f"Successfully loaded {len(self.intervals)} context intervals into binary search engine.")
        except Exception as e:
            self.logger.error(f"Failed to parse session_manifest: {e}")
            self.intervals = []
            self.starts = []

    def lookup(self, timestamp_ms: float):
        """
        Executes a highly performant O(log N) interval search using native python bisect.
        Returns: (context_phase: str, question_id: int, phase_elapsed_ms: float)
        """
        if not self.starts:
            return np.nan, -1, np.nan
        
        # bisect_right finds the insertion point that comes *after* any existing entries 
        # that are <= timestamp_ms. So idx - 1 is the latest interval that started before or exactly at timestamp_ms.
        idx = bisect.bisect_right(self.starts, timestamp_ms) - 1
        
        if idx >= 0:
            start_ms, end_ms, phase_label, question_id = self.intervals[idx]
            
            # Check if the timestamp actually falls strictly inside this specific interval window
            if timestamp_ms < end_ms:
                phase_elapsed_ms = float(timestamp_ms - start_ms)
                return phase_label, question_id, phase_elapsed_ms

        return np.nan, -1, np.nan
