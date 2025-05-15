# Data preparation
import os
import mne
import torch
import tqdm
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Base dataset path
dataset_fold = "/content/drive/MyDrive/tmp/PhysioNetP300/"
train_fold = os.path.join(dataset_fold, "TrainFold")
test_fold = os.path.join(dataset_fold, "TestFold")

# Create main directories
os.makedirs(train_fold, exist_ok=True)
os.makedirs(test_fold, exist_ok=True)

all_chans = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
fmin = 0
fmax = 120
tmin = -0.1
tmax = 2

for sub in [1, 2, 3, 4, 5, 6, 7, 9, 11]:  # subjects 8, 10, and 12 were removed
    path = "/content/physionet.org/files/erpbci/1.0.0/s{:02d}".format(sub)
    os.makedirs(path, exist_ok=True)

    for file in os.listdir(path):
        if not file.endswith(".edf"):
            continue

        raw = mne.io.read_raw_edf(os.path.join(path, file))
        raw.pick_channels(all_chans)

        events, event_id = mne.events_from_annotations(raw)
        event_map = {}
        tgt = None

        for k,v in event_id.items():
            if k[0:4] == '#Tgt':
                tgt = k[4]
            event_map[v] = k

        assert tgt is not None
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, event_repeated='drop', preload=True, proj=False)
        epochs.filter(fmin, fmax, method='iir')
        epochs.resample(256)
        stims = [x[2] for x in epochs.events]
        data = epochs.get_data()

        # Create lists to store valid samples and their labels
        valid_samples = []
        valid_labels = []

        for i, (d, t) in enumerate(zip(data, stims)):
            t = event_map[t]
            if t.startswith('#Tgt') or t.startswith('#end') or t.startswith('#start') or t[0] == '#':
                continue
            label = 1 if tgt in t else 0
            valid_samples.append(d)
            valid_labels.append(label)

        # Convert to numpy arrays for easier manipulation
        valid_samples = np.array(valid_samples)
        valid_labels = np.array(valid_labels)

        # Create indices for train-test split
        n_samples = len(valid_samples)
        indices = np.random.permutation(n_samples)
        train_size = int(0.8 * n_samples)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # Save train samples
        for idx in train_indices:
            x = torch.tensor(valid_samples[idx] * 1e3)
            y = valid_labels[idx]
            spath = os.path.join(train_fold, f'{y}/')
            os.makedirs(spath, exist_ok=True)
            spath = os.path.join(spath, f'{idx}.sub{sub}')
            torch.save(x, spath)

        # Save test samples
        for idx in test_indices:
            x = torch.tensor(valid_samples[idx] * 1e3)
            y = valid_labels[idx]
            spath = os.path.join(test_fold, f'{y}/')
            os.makedirs(spath, exist_ok=True)
            spath = os.path.join(spath, f'{idx}.sub{sub}')
            torch.save(x, spath) 