# EEGUnity: Supporting Multi-Modal Data

EEGUnity supports the classification of channels into the following types:

- `EEG` — Electroencephalography  
- `EOG` — Electrooculography  
- `EMG` — Electromyography  
- `MEG` — Magnetoencephalography  
- `ECG` — Electrocardiography  
- `Dual` — Bipolar channels (e.g., `CFC5-POz`)  
- `Unknown` — Unrecognized or custom signal channels (e.g., `1`, `IBS`, `Blood`)

## Special Notes

- **Dual Channels**:  
  Bipolar channels are marked as `Dual` when EEGUnity is set to **bipolar mode**, which is commonly used in tasks like sleep stage detection. These channels represent differential measurements between two electrode sites.

- **Unknown Channels**:  
  Channels that do not match any recognized naming convention are classified as `Unknown`. This includes numeric or custom-named channels (e.g., `1`, `IBS`, `Blood`). By default, they are treated as `Dual`, but users can reassign their types as needed using preprocessing scripts.

## Multimodal Support

EEGUnity’s multimodal capabilities are tightly integrated with MNE-Python. The level of multimodal support depends on MNE's ability to handle different channel types. Currently, most built-in EEGUnity functions are optimized for EEG data but are also compatible with signals that have similar data structures (e.g., EOG, EMG).

For modalities that are less standardized or not fully supported by default (e.g., ECG, MEG, fNIRS, or custom physiological sensors), users can still apply EEGUnity’s flexible architecture.

## Custom Batch Processing

For full control over preprocessing and analysis workflows across modalities, EEGUnity provides:

```python
unified_dataset.eeg_batch.batch_process()
```

This function enables users to define **customized processing pipelines** tailored to the specific requirements of their modality, allowing for preprocessing steps, filtering, artifact removal, segmentation, and feature extraction — even on non-EEG data.

## Recommendations

- Ensure all modalities are **properly labeled** in the raw data files or during the channel mapping phase.
- Use `unified_dataset.map_channel_types()` to automatically classify or manually adjust channel types.
- For large-scale multimodal projects, consider organizing channels into **modality-specific groups** to simplify downstream analysis.

EEGUnity aims to offer **unified, scalable, and reproducible** workflows for diverse neural and physiological signals in research and real-world applications.
ng to meet their requiqrement