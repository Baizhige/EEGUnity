"""BCIC IV 2a stage1 kernel for EEGUnity.

This kernel updates metadata and events in-memory without saving new files.
It is intended to be loaded externally by EEGUnity.

Spec example
------------
"/path/to/bcic_iv_2a_kernel.py"
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import mne
import numpy as np
import scipy.io as scio


_SUBJECT_DICT: Dict[str, Dict[str, Any]] = {
    "A01": {"sex": "female", "age": 22},
    "A02": {"sex": "female", "age": 24},
    "A03": {"sex": "male", "age": 26},
    "A04": {"sex": "female", "age": 24},
    "A05": {"sex": "male", "age": 24},
    "A06": {"sex": "female", "age": 23},
    "A07": {"sex": "male", "age": 25},
    "A08": {"sex": "male", "age": 23},
    "A09": {"sex": "male", "age": 17},
}


@dataclass
class BCICIV2aKernel:
    """Stage1 kernel for the BCIC IV 2a dataset."""

    KERNEL_ID: str = "bcic-iv-2a-stage1"

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
        subject_id = os.path.basename(file_path)[:3]
        if subject_id not in _SUBJECT_DICT:
            raise ValueError(f"Unrecognized subject id: {subject_id}")

        # 1) Write subject + device meta into raw.info['description'].
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "unknown",
                "cap": "Ag/AgCl",
                "age": _SUBJECT_DICT[subject_id]["age"],
                "sex": _SUBJECT_DICT[subject_id]["sex"],
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # 2) Rename channels + montage + eog types.
        raw.rename_channels(
            {
                "EEG-Fz": "Fz",
                "EEG-0": "FC3",
                "EEG-1": "FC1",
                "EEG-2": "FCz",
                "EEG-3": "FC2",
                "EEG-4": "FC4",
                "EEG-5": "C5",
                "EEG-C3": "C3",
                "EEG-6": "C1",
                "EEG-Cz": "Cz",
                "EEG-7": "C2",
                "EEG-C4": "C4",
                "EEG-8": "C6",
                "EEG-9": "CP3",
                "EEG-10": "CP1",
                "EEG-11": "CPz",
                "EEG-12": "CP2",
                "EEG-13": "CP4",
                "EEG-14": "P1",
                "EEG-15": "Pz",
                "EEG-16": "P2",
                "EEG-Pz": "POz",
            }
        )

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.info.set_montage(montage, on_missing="ignore")
        raw.set_channel_types(
            {"EOG-left": "eog", "EOG-central": "eog", "EOG-right": "eog"}
        )

        # 3) Re-map events.
        event_id = {
            "Rejected trial": 1,
            "Eye movements": 2,
            "Idling EEG (eyes open)": 3,
            "Idling EEG (eyes closed)": 4,
            "Start of a new run": 5,
            "Start of a trial": 6,
            "Cue onset left (class 1)": 7,
            "Cue onset right (class 2)": 8,
            "Cue onset foot (class 3)": 9,
            "Cue onset tongue (class 4)": 10,
        }

        events, original_event_id = mne.events_from_annotations(raw)

        for desc, new_id in event_id.items():
            if desc in original_event_id:
                old_id = original_event_id[desc]
                events[events[:, 2] == old_id, 2] = new_id

        # 4) If file ends with 'E.gdf', use adjacent .mat classlabel to refine events.
        base, ext = os.path.splitext(file_path)
        if base.endswith("E") and ext.lower() == ".gdf":
            mat_filepath = f"{base}.mat"
            if os.path.exists(mat_filepath):
                mat_data = scio.loadmat(mat_filepath)
                values_from_mat = mat_data["classlabel"].flatten() + 6
                replacement_indices = np.where(events[:, 2] == 7)[0]
                if len(replacement_indices) >= len(values_from_mat):
                    events[replacement_indices[: len(values_from_mat)], 2] = values_from_mat
                else:
                    raise ValueError(
                        f"Not enough cue events to replace using {mat_filepath}."
                    )

        # 5) Convert back to annotations (descriptions are from the mapping).
        id_to_desc = {v: k for k, v in event_id.items()}
        annotations = mne.annotations_from_events(
            events=events,
            sfreq=raw.info["sfreq"],
            event_desc=id_to_desc,
        )
        raw.set_annotations(annotations)
        return raw


KERNEL = BCICIV2aKernel()