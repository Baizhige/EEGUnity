"""Figshare LargeMI stage1 kernel for EEGUnity.

This kernel parses marker sequences from the .mat file and constructs MNE
annotations. It also updates raw.info['description'].

Spec example
------------
"/path/to/figshare_largemi_kernel.py"
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import mne
import scipy.io as sio


_SUBJECT_DICT: Dict[str, Dict[str, Any]] = {
    "A": {"sex": "Male", "Age": "20-25"},
    "B": {"sex": "Male", "Age": "20-25"},
    "C": {"sex": "Male", "Age": "25-30"},
    "D": {"sex": "Male", "Age": "25-30"},
    "E": {"sex": "Female", "Age": "20-25"},
    "F": {"sex": "Male", "Age": "30-35"},
    "G": {"sex": "Male", "Age": "30-35"},
    "H": {"sex": "Male", "Age": "20-25"},
    "I": {"sex": "Female", "Age": "20-25"},
    "J": {"sex": "Female", "Age": "20-25"},
    "K": {"sex": "Male", "Age": "20-25"},
    "L": {"sex": "Female", "Age": "20-25"},
    "M": {"sex": "Female", "Age": "20-25"},
}


@dataclass
class FigshareLargeMIKernel:
    """Stage1 kernel for the figshare-largemi dataset."""

    KERNEL_ID: str = "figshare-largemi-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        """Apply stage1 preprocessing to a single record.

        Parameters
        ----------
        udataset
            EEGUnity dataset context (shared-attr holder).
        raw
            In-memory MNE Raw object.
        row
            Locator row (pandas.Series-like) with at least ``File Path``.

        Returns
        -------
        mne.io.BaseRaw
            The modified raw object.
        """
        file_path = row["File Path"]
        file_name = os.path.basename(file_path)

        mat = sio.loadmat(file_path, simplify_cells=True)
        marker = mat["o"]["marker"]

        # Infer subject id from file name: "...subjectX..."
        idx = file_name.find("subject")
        if idx == -1 or idx + len("subject") >= len(file_name):
            raise ValueError(f"Cannot infer subject id from file name: {file_name}")
        subject_id = file_name[idx + len("subject")]
        subject_key = subject_id.upper()
        if subject_key not in _SUBJECT_DICT:
            raise ValueError(f"Unknown subject id: {subject_id}")

        # Event dictionary depends on task family.
        if file_name.startswith(("CLA", "HaLT", "FREEDOM")):
            event_dict = {
                1: "left hand MI",
                2: "right hand MI",
                3: "passive state",
                4: "left leg MI",
                5: "tongue MI",
                6: "right leg MI",
            }
        elif file_name.startswith("5F"):
            event_dict = {
                1: "thumb MI",
                2: "index finger MI",
                3: "middle finger MI",
                4: "ring finger MI",
                5: "pinkie finger MI",
            }
        else:
            raise ValueError(f"Unrecognized task prefix in file name: {file_name}")

        # Build annotations by tracking contiguous marker segments.
        sfreq = raw.info["sfreq"]
        annotations = []
        current_event = None
        current_start = 0

        for i, code in enumerate(marker):
            if code not in event_dict:
                code = 0

            if code != 0:
                if current_event is None:
                    current_event = code
                    current_start = i
                elif current_event != code:
                    onset = current_start / sfreq
                    duration = (i - current_start) / sfreq
                    annotations.append((onset, duration, event_dict[current_event]))
                    current_event = code
                    current_start = i
            else:
                if current_event is not None:
                    onset = current_start / sfreq
                    duration = (i - current_start) / sfreq
                    annotations.append((onset, duration, event_dict[current_event]))
                    current_event = None

        if current_event is not None:
            onset = current_start / sfreq
            duration = (len(marker) - current_start) / sfreq
            annotations.append((onset, duration, event_dict[current_event]))

        if annotations:
            onset, duration, desc = zip(*annotations)
            raw.set_annotations(mne.Annotations(list(onset), list(duration), list(desc)))
        else:
            raw.set_annotations(mne.Annotations([], [], []))

        # Update description.
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "Nihon Kohden EEG-1200 device",
                "cap": "Nihon Kohden EEG-1200 device",
                "age": _SUBJECT_DICT[subject_key]["Age"],
                "sex": _SUBJECT_DICT[subject_key]["sex"],
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        return raw


KERNEL = FigshareLargeMIKernel()