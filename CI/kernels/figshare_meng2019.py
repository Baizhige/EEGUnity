"""Figshare Meng2019 stage1 kernel for EEGUnity.

This kernel generates annotations from TargetCode/ResultCode transitions
and updates raw.info['description'] depending on experiment folder.

Spec example
------------
"/path/to/figshare_meng2019_kernel.py"
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne
import scipy.io as sio


@dataclass
class FigshareMeng2019Kernel:
    """Stage1 kernel for the figshare-meng2019 dataset."""

    KERNEL_ID: str = "figshare-meng2019-stage1"

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
        mat = sio.loadmat(file_path, simplify_cells=True)

        target_stim = mat["Experimental_states"]["TargetCode"]
        result_stim = mat["Experimental_states"]["ResultCode"]

        target_annotations = []
        result_annotations = []

        last_target = None
        for i, stim in enumerate(target_stim):
            if stim != last_target:
                if stim == 1:
                    target_annotations.append((i, "target cursor plus", 0))
                elif stim == 2:
                    target_annotations.append((i, "target cursor minus", 0))
                elif stim == 0:
                    result_annotations.append((i, "target rest", 0))
            last_target = stim

        last_result = None
        for i, stim in enumerate(result_stim):
            if stim != last_result:
                if stim == 1:
                    result_annotations.append((i, "result cursor plus", 0))
                elif stim == 2:
                    result_annotations.append((i, "result cursor minus", 0))
            last_result = stim

        annotations = sorted(target_annotations + result_annotations, key=lambda x: x[0])

        sfreq = raw.info["sfreq"]
        onset_seconds = [a[0] / sfreq for a in annotations]
        durations = [a[2] for a in annotations]
        labels = [a[1] for a in annotations]
        raw.set_annotations(mne.Annotations(onset_seconds, durations, labels))

        experiment_tag = os.path.basename(os.path.dirname(file_path))
        if experiment_tag in ("Exp1", "Exp2"):
            eeg_desc = {
                "amplifier": "Neuroscan SynAmps RT",
                "cap": "unknown",
                "age": "18-50",
                "sex": "unknown",
                "handedness": "unknown",
            }
        elif experiment_tag == "Exp3":
            eeg_desc = {
                "amplifier": "Biosemi Active Two EEG systems",
                "cap": "unknown",
                "age": "18-50",
                "sex": "unknown",
                "handedness": "unknown",
            }
        else:
            raise ValueError(f"Unrecognized experiment folder: {experiment_tag}")

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": eeg_desc,
        }
        raw.info["description"] = json.dumps(description_dict)

        return raw


KERNEL = FigshareMeng2019Kernel()