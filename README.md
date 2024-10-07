# CAUEEG Dataset

This is a curated version of the [CAUEEG Dataset](https://github.com/ipis-mjkim/caueeg-dataset) in BIDS format.



- The serial numbers were mapped to participant IDs. Note that this is not ideal because serial numbers in the CAUEEG source data correspond to separate EEG recordings rather than participants. CAUEEG contains 1379 EEG recordings acquired from 1155 patients, thus some of the recordings mapped to different participant IDs here will actually correspond to the same participant. The dataset authors have not shared participant identifiers. For evaluation purposes, the authors provide an official train-validation-test split where overlapping participants were excluded from the validation and test sets (see below).
- Labels were added to the participants.tsv file (normality_label column containing {"normal", "abnormal"} and the dementia_label column containing {"normal", "mci", "dementia", NaN}). Note that the normal vs abnormal classification used in the CAUEEG dataset is based on the clinical diagnosis information and indicates the absence of any clinical diagnoses. This is very different from the classification of EEG recordings as normal or abnormal as used in datasets like TUAB.
- The official splits for the dementia (normal vs mci vs dementia) and normality (normal vs abnormal) prediction tasks were added to the participants.tsv file. Note that the authors provided two different splits for each task: one where some participants overlap between train, val and test splits and one where overlapping participants were removed from the validation and test data. Both versions were added in the columns \*_split and \*_split_no_overlap respectively.
- Event information was extracted from the event json files. Most events, including eyes open/closed, photic stimulation, hyperventilation, as well as various annotations like eye blinks, sweating, drowsy, and artifacts were parsed, but some (e.g. for electrode impedance checks) were discarded. Note that inconsistencies between the json event files and corresponding edf data files were observed (events outside the duration of the data) for 199 files, so events may not be reliable.
- Channel names and types were harmonised and mapped to the new 10-20 nomenclature (T3 renamed to T7 etc.).
- Channel locations were set from the standard 10-20 montage.




## Reproduce from the Source Data

1. Clone repository

   ```
   git clone https://github.com/bomatter/data-CAUEEG.git CAUEEG
   cd CAUEEG
   ```

2. Install dependencies or use an existing environment with mne, mne-bids, and openpyxl installed.
   Example using mamba:

   ```
   mamba create -n bidsify python=3.10 mne>=1.8 mne-bids openpyxl
   mamba activate bidsify
   ```

3. (optional) Modify or delete the `.gitignore` file to start tracking the data folders with [DataLad](https://www.datalad.org/).

   ```
   mamba install datalad
   rm .gitignore
   datalad save -m"start tracking data folders"
   ```

4. Request access and download the data to `sourcedata/`

   Save if you are using DataLad to track the data folders:

   ```
   datalad save -m"downloaded sourcedata"
   ```

5. Run the BIDS conversion script.

   ```
   python code/convert_CAUEEG_to_BIDS.py
   ```

   or using DataLad:

   ```
   datalad run \
      -m "run data curation" \
      -i "sourcedata/*" \
      -o "rawdata/*" \
      -o "derivatives/*" \
      "python code/convert_CAUEEG_to_BIDS.py"
   ```