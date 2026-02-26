import os
import re
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample
import stft
import matplotlib.pyplot as plt
from mne.io import read_raw_edf
from utils.save_load import (
    save_hickle_file,
    load_hickle_file,
)

SZ_KEY = "(?i)sz"
NSZ_KEY = "(?i)bckg"


def load_signals_CHBMIT(data_dir, patient, data_type, settings, verbose=False):
    if "MIT" in data_dir:
        dataset = "CHB-MIT"
    elif "Siena" in data_dir:
        dataset = "CHB-Siena"
    else:
        raise ValueError("Unknown dataset in data_dir")
    if dataset == "CHB-MIT" and patient == 21:
        pat_id = "sub-01"
    else:
        pat_id = f"sub-{patient.zfill(2)}"

    print(f"Loading {data_type} data for patient {patient} from {dataset} dataset...")
    data_df = pd.read_csv(settings["combined_csv"])
    if patient == 21:
        data_df = data_df[
            (data_df["subject"] == "sub-01") & (data_df["ses"] == "ses-02")
        ]
    else:
        data_df = data_df[(data_df["subject"] == pat_id) & (data_df["ses"] == "ses-01")]
    if data_type == "ictal":
        data_df = data_df[data_df["eventType"].str.contains(SZ_KEY)]
    elif data_type == "interictal":
        data_df = data_df[data_df["eventType"].str.contains(NSZ_KEY)]
    else:
        raise ValueError("data_type must be either 'ictal' or 'interictal'")
    data_df["fileName"] = data_df["fileName"].str.replace("events.tsv", "eeg.edf")
    edf_list = data_df["fileName"].tolist()
    for edf_file in sorted(set(edf_list)):
        rawEEG = read_raw_edf(edf_file, verbose=False, preload=True)
        eegData, eegTimes = rawEEG[:]
        if data_type == "ictal":
            indices = [
                idx for idx, x in enumerate(data_df["fileName"]) if x == edf_file
            ]
            run_match = re.search(r"(run-)(\d+)", os.path.basename(edf_file))
            if run_match:
                run_prefix = run_match.group(1)
                run_number = run_match.group(2)
                run_prev = f"{run_prefix}{str(int(run_number)-1).zfill(2)}"
                prev_file = re.sub(r"(run-)\d+", run_prev, edf_file)
            else:
                prev_file = None
                if verbose:
                    print(f"Run number not found in {edf_file}")

            if len(indices) > 0:
                print("%d seizures in the file %s" % (len(indices), edf_file))
                prev_sp = -1e6
                for i in range(len(indices)):
                    st = int(data_df.iloc[indices[i]]["onset"] * settings["sampfreq"])
                    dur = int(
                        data_df.iloc[indices[i]]["duration"] * settings["sampfreq"]
                    )
                    sp = st + dur
                    SOP = int(settings["SOPsec"] * settings["sampfreq"])
                    if st - SOP > prev_sp:
                        prev_sp = sp
                        if st - SOP >= 0:
                            data = eegData[:, st - SOP : st]
                        else:
                            if os.path.exists(prev_file):
                                prev_rawEEG = read_raw_edf(
                                    prev_file, verbose=False, preload=True
                                )
                                prev_eegData, _ = prev_rawEEG[:]
                                # print(
                                #     f"Current file shape: {eegData.shape}, Previous file shape: {prev_eegData.shape}  "
                                # )
                                if st > 0:
                                    data = np.concatenate(
                                        (prev_eegData[:, st - SOP :], eegData[:, :st]),
                                        axis=1,
                                    )
                                else:
                                    data = prev_eegData[:, st - SOP : st]
                            else:  # Handling for run-00 where previous file does not exist
                                if st > 0:
                                    data = eegData[:, :st]
                                else:
                                    if verbose:
                                        print(f"File {edf_file} is not long enough")
                                    continue
                    else:
                        prev_sp = sp
                        continue
                    if verbose:
                        print(
                            f"Data shape: {data.shape}, Start: {st/settings['sampfreq']}s, Duration: {dur/settings['sampfreq']}s"
                        )
                        yield (data.T)
                    else:
                        continue
        elif data_type == "interictal":
            data = eegData
            if verbose:
                print(f"Data shape: {data.shape}")
            yield (data.T)
        else:
            raise ValueError("data_type must be either 'ictal' or 'interictal'")


class PrepDataStudentBIDS:
    def __init__(self, patient, data_type, settings, verbose=False):
        self.patient = patient
        self.settings = settings
        self.type = data_type
        self.freq = self.settings["sampfreq"]
        self.window_sec = self.settings["windowsec"]
        self.window_len = int(self.freq * self.window_sec)
        self.verbose = verbose
        df_sampling = pd.read_csv(self.settings["sampling_csv"])
        self.ictal_ovl_pt = df_sampling[
            df_sampling.Subject == int(self.patient)
        ].ictal_ovl.values[0]
        self.ictal_ovl_len = int(
            self.freq * self.ictal_ovl_pt * self.window_sec
        )  # Step size for overlaps

    def read_raw_signal(self):
        return load_signals_CHBMIT(
            self.settings["datadir"],
            self.patient,
            self.type,
            self.settings,
            self.verbose,
        )

    def preprocess(self, data_gen):

        data = self._process_raw_data_(data_gen)
        return data

    def _process_raw_data_(self, mat_data):
        print("Loading data")
        numts = self.window_sec
        X = []
        y = []

        for data in mat_data:
            if self.type == "ictal":
                y_value = 1
            else:
                y_value = 0

            X_temp = []
            y_temp = []

            totalSample = int(data.shape[0] / self.freq / numts) + 1
            window_len = int(self.freq * numts)
            for i in range(totalSample):
                if (i + 1) * window_len <= data.shape[0]:
                    s = data[i * window_len : (i + 1) * window_len, :]

                    stft_data = stft.spectrogram(
                        s, framelength=self.freq, centered=False
                    )
                    stft_data = np.abs(stft_data) + 1e-6
                    stft_data = np.log10(stft_data)
                    indices = np.where(stft_data <= 0)
                    stft_data[indices] = 0

                    stft_data = np.transpose(stft_data, (2, 1, 0))
                    stft_data = np.concatenate(
                        (
                            stft_data[:, :, 1:57],
                            stft_data[:, :, 64:117],
                            stft_data[:, :, 124:],
                        ),
                        axis=-1,
                    )

                    stft_data = stft_data.reshape(
                        -1,
                        1,
                        stft_data.shape[0],
                        stft_data.shape[1],
                        stft_data.shape[2],
                    )

                    X_temp.append(stft_data)
                    y_temp.append(y_value)

            # overlapped window
            if self.type == "ictal":
                i = 1
                if self.verbose:
                    print("ictal length =", self.ictal_ovl_len)
                while window_len + (i + 1) * self.ictal_ovl_len <= data.shape[0]:
                    s = data[
                        i * self.ictal_ovl_len : i * self.ictal_ovl_len + window_len, :
                    ]

                    stft_data = stft.spectrogram(
                        s, framelength=self.freq, centered=False
                    )
                    stft_data = np.abs(stft_data) + 1e-6
                    stft_data = np.log10(stft_data)
                    indices = np.where(stft_data <= 0)
                    stft_data[indices] = 0

                    stft_data = np.transpose(stft_data, (2, 1, 0))
                    stft_data = np.concatenate(
                        (
                            stft_data[:, :, 1:57],
                            stft_data[:, :, 64:117],
                            stft_data[:, :, 124:],
                        ),
                        axis=-1,
                    )

                    stft_data = stft_data.reshape(
                        -1,
                        1,
                        stft_data.shape[0],
                        stft_data.shape[1],
                        stft_data.shape[2],
                    )

                    X_temp.append(stft_data)
                    # to differentiate between non overlapped and overlapped
                    # samples. Testing only uses non overlapped ones.
                    y_temp.append(2)
                    i += 1

            try:
                X_temp = np.concatenate(X_temp, axis=0)
                y_temp = np.array(y_temp)
                X.append(X_temp)
                y.append(y_temp)
            except:
                print("seizure too short")

        # y = np.array(y)
        print("X", len(X), X[0].shape, "y", len(y), y[0].shape)
        return X, y

    def apply(self):
        # Cache check
        cachedir = self.settings["cachedir"]
        if os.path.exists(cachedir) == False:
            os.makedirs(cachedir)
            print(f"Created cache directory: {cachedir}")
        filename = f"{self.type}_{self.patient}.hkl"
        cache_path = os.path.join(cachedir, filename)
        cache = load_hickle_file(cache_path)

        if cache is not None:
            if self.verbose:
                print("Loaded from cache")
            return cache

        data = self.read_raw_signal()
        X, y = self.preprocess(data)
        save_hickle_file(cache_path, [X, y])
        if self.verbose:
            print(f"Saved to cache: {cache_path}")
        return X, y
