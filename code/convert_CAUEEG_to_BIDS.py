import os
import warnings
import json
import shutil
import pandas as pd

from tqdm import tqdm
from pathlib import Path

import mne
from mne_bids import write_raw_bids, BIDSPath

# Suppress known warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Converting data files to BrainVision.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message='.*Encountered data in "int" format.*')
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*annotation\(s\) that were expanding outside the data range.*")


channel_name_mapping = {
    'Fp1-AVG': 'Fp1',
    'F3-AVG': 'F3',
    'C3-AVG': 'C3',
    'P3-AVG': 'P3',
    'O1-AVG': 'O1',
    'Fp2-AVG': 'Fp2',
    'F4-AVG': 'F4',
    'C4-AVG': 'C4',
    'P4-AVG': 'P4',
    'O2-AVG': 'O2',
    'F7-AVG': 'F7',
    'T3-AVG': 'T3',
    'T5-AVG': 'T5',
    'F8-AVG': 'F8',
    'T4-AVG': 'T4',
    'T6-AVG': 'T6',
    'FZ-AVG': 'Fz',
    'CZ-AVG': 'Cz',
    'PZ-AVG': 'Pz',
    'EKG': 'ECG',
    'Photic': 'Photic'
}


channel_type_mapping = {
    'Fp1-AVG': 'eeg',
    'F3-AVG': 'eeg',
    'C3-AVG': 'eeg',
    'P3-AVG': 'eeg',
    'O1-AVG': 'eeg',
    'Fp2-AVG': 'eeg',
    'F4-AVG': 'eeg',
    'C4-AVG': 'eeg',
    'P4-AVG': 'eeg',
    'O2-AVG': 'eeg',
    'F7-AVG': 'eeg',
    'T3-AVG': 'eeg',
    'T5-AVG': 'eeg',
    'F8-AVG': 'eeg',
    'T4-AVG': 'eeg',
    'T6-AVG': 'eeg',
    'FZ-AVG': 'eeg',
    'CZ-AVG': 'eeg',
    'PZ-AVG': 'eeg',
    'EKG': 'ecg',
    'Photic': 'stim'
}


def determine_dementia_type(row):
    if pd.notna(row['ad']) and row['ad']:
        return 'ad'
    elif pd.notna(row['vd']) and row['vd']:
        return 'vd'
    elif pd.notna(row['ad_vd_mixed']) and row['ad_vd_mixed']:
        return 'ad_vd_mixed'
    elif pd.notna(row['ftd']) and row['ftd']:
        return 'ftd'
    elif pd.notna(row['parkinson_dementia']) and row['parkinson_dementia']:
        return 'parkinson_dementia'
    else:
        return None
    

def determine_dementia_label(row):
    normal = row['normal'] if pd.notna(row['normal']) else False
    mci = row['mci'] if pd.notna(row['mci']) else False
    dementia = row['dementia'] if pd.notna(row['dementia']) else False

    if normal:
        assert not mci and not dementia
        return 'normal'
    elif mci:
        assert not normal and not dementia
        return 'mci'
    elif dementia:
        assert not normal and not mci
        return 'dementia'
    else:
        return None
    

def parse_events(events, sfreq):
    onsets = []
    durations = []
    descriptions = []

    # Check that the events are ordered by onset time
    assert (events["onset"].diff().dropna() >= 0).all(), "The events are not ordered by onset time"

    for i, event in events.iterrows():
        if event["description"] in ["Eyes Open", "Eyes Closed"]:
            
            onset = event["onset"] / sfreq

            # Find the offset as the next event that is either Eyes Open, Eyes Closed, or Paused
            for j in range(i + 1, len(events)):
                if events.iloc[j]["description"] in ["Eyes Open", "Eyes Closed", "Paused"]:
                    offset = events.iloc[j]["onset"] / sfreq
                    break
            
            onsets.append(onset)
            durations.append(offset - onset)
            descriptions.append(event["description"])

        elif event["description"].startswith("Photic On"):
            onset = event["onset"] / sfreq

            # Find the offset as the next event that is either Photic Off or Paused
            for j in range(i + 1, len(events)):
                if events.iloc[j]["description"] in ["Photic Off", "Paused"]:
                    offset = events.iloc[j]["onset"] / sfreq
                    break

            onsets.append(onset)
            durations.append(offset - onset)
            descriptions.append(event["description"])

        elif event["description"].startswith("HV") and event["description"].endswith("On"):
            onset = event["onset"] / sfreq

            # Find the offset as the next event that is either "HV - Off" or Paused
            for j in range(i + 1, len(events)):
                if events.iloc[j]["description"] in ["HV - Off", "Paused"]:
                    offset = events.iloc[j]["onset"] / sfreq
                    break

            onsets.append(onset)
            durations.append(offset - onset)
            descriptions.append(event["description"])
        
        elif "drowsy" in event["description"].lower():
            onsets.append(event["onset"] / sfreq)
            durations.append(0)
            descriptions.append("drowsy")

        elif (
            "cough" in event["description"].lower() or
            'couch' in event["description"].lower()
        ):
            onsets.append(event["onset"] / sfreq)
            durations.append(0)
            descriptions.append("cough")

        elif "chew" in event["description"].lower():
            onsets.append(event["onset"] / sfreq)
            durations.append(0)
            descriptions.append("chewing")
        
        elif "sweat" in event["description"].lower():
            onsets.append(event["onset"] / sfreq)
            durations.append(0)
            descriptions.append("sweating")

        elif "blink" in event["description"].lower():
            onsets.append(event["onset"] / sfreq)
            durations.append(0)
            descriptions.append("eye blink")

        elif "eye movement" in event["description"].lower():
            onsets.append(event["onset"] / sfreq)
            durations.append(0)
            descriptions.append("eye movement")

        elif (
            "move" in event["description"].lower() or
            'jerk' in event["description"].lower()
        ):
            onsets.append(event["onset"] / sfreq)
            durations.append(0)
            descriptions.append("movement")

        elif "seizure" in event["description"].lower():
            onsets.append(event["onset"] / sfreq)
            durations.append(0)
            descriptions.append("seizure")

        elif "artifact" in event["description"].lower():
            onsets.append(event["onset"] / sfreq)
            durations.append(0)
            descriptions.append("artifact")

    return onsets, durations, descriptions


def convert_caueeg_to_bids():
    current_dir = Path(__file__).parents[1]
    os.chdir(current_dir)

    sourcedata_dir = Path("sourcedata/caueeg-dataset")
    rawdata_dir = Path("rawdata")  # Where the BIDSified data should be written to

    # Delete rawdata directory if it already exists to ensure a clean slate
    if rawdata_dir.exists():
        print(f"Deleting existing rawdata directory: {rawdata_dir}")
        shutil.rmtree(rawdata_dir)


    # Load annotations and create participants.tsv from it
    participants_tsv = pd.read_excel(sourcedata_dir / "annotation.xlsx", dtype={'serial': str})

    # Convert the boolean columns to boolean type
    boolean_columns = [
        'dementia', 'ad', 'load', 'eoad', 'vd', 'sivd', 'ad_vd_mixed', 'mci', 'mci_ad', 'mci_amnestic',
        'mci_amnestic_ef', 'mci_amnestic_rf', 'mci_non_amnestic', 'mci_multi_domain', 'mci_vascular',
        'normal', 'cb_normal', 'smi', 'hc_normal', 'ftd', 'bvftd', 'language_ftd', 'semantic_aphasia',
        'non_fluent_aphasia', 'parkinson_synd', 'parkinson_disease', 'parkinson_dementia', 'nph', 'tga'
    ]

    participants_tsv[boolean_columns] = participants_tsv[boolean_columns].astype('boolean')
    participants_tsv.rename(columns={'serial': 'participant_id'}, inplace=True)  # user serial as participant_id

    # Add additional information to participants.tsv
    participants_tsv['dementia_type'] = participants_tsv.apply(determine_dementia_type, axis=1)
    participants_tsv['dementia_label'] = participants_tsv.apply(determine_dementia_label, axis=1)
    participants_tsv['normality_label'] = participants_tsv["normal"].apply(lambda x: 'normal' if (pd.notna(x) and x) else 'abnormal')

    # Add split information for dementia prediction
    with open(sourcedata_dir / "dementia.json") as f:
        dementia_split = json.load(f)

    id_to_dementia_split = {e["serial"]: "train" for e in dementia_split["train_split"]}
    id_to_dementia_split.update({e["serial"]: "val" for e in dementia_split["validation_split"]})
    id_to_dementia_split.update({e["serial"]: "test" for e in dementia_split["test_split"]})

    participants_tsv["dementia_split"] = participants_tsv["participant_id"].map(id_to_dementia_split)


    # Version of the split without participant overlaps between train, val, and test
    with open(sourcedata_dir / "dementia-no-overlap.json") as f:
        dementia_split_no_overlap = json.load(f)

    id_to_dementia_split_no_overlap = {e["serial"]: "train" for e in dementia_split_no_overlap["train_split"]}
    id_to_dementia_split_no_overlap.update({e["serial"]: "val" for e in dementia_split_no_overlap["validation_split"]})
    id_to_dementia_split_no_overlap.update({e["serial"]: "test" for e in dementia_split_no_overlap["test_split"]})

    participants_tsv["dementia_split_no_overlap"] = participants_tsv["participant_id"].map(dementia_split_no_overlap)


    # Add split information for normality prediction (abnormal vs. normal)
    with open(sourcedata_dir / "abnormal.json") as f:
        abnormal_split = json.load(f)

    id_to_abnormal_split = {e["serial"]: "train" for e in abnormal_split["train_split"]}
    id_to_abnormal_split.update({e["serial"]: "val" for e in abnormal_split["validation_split"]})
    id_to_abnormal_split.update({e["serial"]: "test" for e in abnormal_split["test_split"]})

    participants_tsv["normality_split"] = participants_tsv["participant_id"].map(id_to_abnormal_split)


    # Version of the split without participant overlaps between train, val, and test
    with open(sourcedata_dir / "abnormal-no-overlap.json") as f:
        abnormal_split_no_overlap = json.load(f)

    id_to_abnormal_split_no_overlap = {e["serial"]: "train" for e in abnormal_split_no_overlap["train_split"]}
    id_to_abnormal_split_no_overlap.update({e["serial"]: "val" for e in abnormal_split_no_overlap["validation_split"]})
    id_to_abnormal_split_no_overlap.update({e["serial"]: "test" for e in abnormal_split_no_overlap["test_split"]})

    participants_tsv["normality_split_no_overlap"] = participants_tsv["participant_id"].map(id_to_abnormal_split_no_overlap)

    # Check that labels extracted from excel and json files are consistent
    assert all([
        e["class_name"].lower() == participants_tsv.loc[participants_tsv.participant_id == e["serial"], 'normality_label'].values[0]
        for e in abnormal_split["train_split"]
    ])

    assert all([
        e["class_name"].lower() == participants_tsv.loc[participants_tsv.participant_id == e["serial"], 'normality_label'].values[0]
        for e in abnormal_split["validation_split"]
    ])

    assert all([
        e["class_name"].lower() == participants_tsv.loc[participants_tsv.participant_id == e["serial"], 'normality_label'].values[0]
        for e in abnormal_split["test_split"]
    ])

    assert all([
        e["class_name"].lower() == participants_tsv.loc[participants_tsv.participant_id == e["serial"], 'dementia_label'].values[0]
        for e in dementia_split["train_split"]
    ])

    assert all([
        e["class_name"].lower() == participants_tsv.loc[participants_tsv.participant_id == e["serial"], 'dementia_label'].values[0]
        for e in dementia_split["validation_split"]
    ])

    assert all([
        e["class_name"].lower() == participants_tsv.loc[participants_tsv.participant_id == e["serial"], 'dementia_label'].values[0]
        for e in dementia_split["test_split"]
    ])

    errors = []
    print("Converting CAUEEG files to BIDS...")
    for _, row in tqdm(participants_tsv.iterrows(), total=len(participants_tsv)):
        try:
            # Load raw data
            raw = mne.io.read_raw_edf(sourcedata_dir / "signal" / "edf" / (row['participant_id'] + ".edf"), verbose=False)
            sfreq = raw.info['sfreq']
            
            # Set channel names and types
            raw.set_channel_types({k: v for k, v in channel_type_mapping.items() if k in raw.ch_names}, on_unit_change="ignore")
            raw.rename_channels({k: v for k, v in channel_name_mapping.items() if k in raw.ch_names})

            # Set channel locations from standard montage
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)

            # Parse event information from json file
            events = pd.read_json(sourcedata_dir / "event" / (row['participant_id'] + ".json"))
            events.columns = ["onset", "description"]
            onsets, durations, descriptions = parse_events(events, sfreq)

            annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions
            )

            raw.set_annotations(annotations)

            # Write raw data to BIDS
            write_raw_bids(
                raw,
                bids_path=BIDSPath(root=rawdata_dir, subject=row["participant_id"], task="rest"),
                overwrite=True,
                verbose=False,
                allow_preload=True,
                format="BrainVision"
            )

        except Exception as e:
            error_msg = (f"Error processing {row['participant_id']}: {e}")
            print(error_msg)
            errors.append(error_msg)
            continue


    # Overwrite participants.tsv
    participants_tsv.to_csv(rawdata_dir / "participants.tsv", sep="\t", index=False)

    # Print final summary
    print(f"BIDS conversion completed. {len(participants_tsv)- len(errors)}/{len(participants_tsv)} files were successfully processed.")
    if errors:
        print("Errors occurred for the following files:")
        print("\n".join(errors))



if __name__ == "__main__":
    convert_caueeg_to_bids()