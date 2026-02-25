# Cooperation with Multiple Computers or Servers using EEGUnity

## Prerequisites

Before using this script, you should have:

1. **At least 2 computers or servers**  
   - EEGUnity toolbox installed on all machines  
   - The same datasets stored on each machine

## Assumptions

Let us assume you have two researchers: A and B, each working on their own computer.

## For Researcher A

Researcher A writes the initial data processing scripts and creates a *locator* to manage the dataset.  
The *locator* specifies which datasets or files to process.  
Only the *locator* (not the raw data) is shared with Researcher B.

## For Researcher B

After receiving the *locator* from A, Researcher B:

1. Uses `unified_dataset.eeg_batch.replace_path` to update file paths according to their local dataset structure.  
   (Alternatively, Researcher B can use external software like Microsoft Excel to edit the locator manually.)

2. Runs `unified_dataset.eeg_batch.get_file_hashes()` to verify that the file hashes match, ensuring the datasets are identical.
