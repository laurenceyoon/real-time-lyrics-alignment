import IPython.display as ipd
import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm
import librosa
from attributedict.collections import AttributeDict
import torch

from .config import (
    FRAME_RATE,
    TOLERANCES,
    FILENAME_PREFIX,
    NOTE_ANN_DIR,
    FEATURES,
    SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    NORM,
    CRNN_MODEL_PATH,
)
from .CRNN_model import CRNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load acoustic model
save_data = torch.load(CRNN_MODEL_PATH, map_location=DEVICE)
model_state_dict = save_data["model_state_dict"]
config = AttributeDict(save_data["config"])
consts = AttributeDict(save_data["consts"])
config.device = DEVICE

crnn_model = CRNN(config, consts)
crnn_model.to("mps")

crnn_model.load_state_dict(model_state_dict, strict=False)
crnn_model.eval()


def softmax_with_temperature(z, T=1):
    y = np.exp(z / T) / np.sum(np.exp(z / T), axis=0)
    return y


def process_chroma(y):
    chroma_stft = librosa.feature.chroma_stft(
        y=y,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        norm=NORM,
        center=False,
    )
    return chroma_stft


def process_mfcc(y):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        center=False,
    )
    return np.log1p(mfcc * 5) / 4


def process_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        norm=NORM,
        n_mels=N_MELS,
        center=False,
    )
    return np.log1p(mel * 5) / 4


def process_phonemes(y):
    x_tensor = torch.from_numpy(y).float().to("mps")

    with torch.no_grad():
        batch = {"audio": x_tensor}
        predictions = crnn_model.run_on_batch(batch, cal_loss=False)

    phonemes = predictions["frame"].squeeze().T.cpu().numpy()  # [39, T]
    phonemes = phonemes[:, 1:-1]  # remove <sos> and <eos>
    phonemes = softmax_with_temperature(phonemes, T=1)
    phonemes = np.log1p(phonemes * 5) / 4
    return phonemes


def compute_strict_alignment_path_mask(P):
    """Compute strict alignment path from a warping path

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        P (list or np.ndarray): Wapring path

    Returns:
        P_mod (list or np.ndarray): Strict alignment path
    """
    P = np.array(P, copy=True)
    N, M = P[-1]
    # Get indices for strict monotonicity
    keep_mask = (P[1:, 0] > P[:-1, 0]) & (P[1:, 1] > P[:-1, 1])
    # Add first index to enforce start boundary condition
    keep_mask = np.concatenate(([True], keep_mask))
    # Remove all indices for of last row or column
    keep_mask[(P[:, 0] == N) | (P[:, 1] == M)] = False
    # Add last index to enforce end boundary condition
    keep_mask[-1] = True
    P_mod = P[keep_mask, :]

    return P_mod


def make_path_strictly_monotonic(P: np.ndarray) -> np.ndarray:
    """Compute strict alignment path from a warping path

    Wrapper around "compute_strict_alignment_path_mask" from libfmp.

    Parameters
    ----------
    P: np.ndarray [shape=(2, N)]
        Warping path

    Returns
    -------
    P_mod: np.ndarray [shape=(2, M)]
        Strict alignment path, M <= N
    """
    P_mod = compute_strict_alignment_path_mask(P.T)

    return P_mod.T


def transfer_note_positions(wp, note_ann_1, feature_rate=FRAME_RATE):
    x, y = wp[0] / feature_rate, wp[1] / feature_rate
    f = scipy.interpolate.interp1d(x, y, kind="linear")
    note_positions_1_transferred_to_2 = f(note_ann_1)
    return note_positions_1_transferred_to_2


def get_stats(
    wp,
    note_ann_filepath_1,
    note_ann_filepath_2,
    feature_rate=FRAME_RATE,
    tolerances=TOLERANCES,
):  # tolerances in milliseconds
    wp = make_path_strictly_monotonic(wp)

    note_ann_1 = pd.read_csv(filepath_or_buffer=note_ann_filepath_1, delimiter=",")[
        "start"
    ]
    note_ann_2 = pd.read_csv(filepath_or_buffer=note_ann_filepath_2, delimiter=",")[
        "start"
    ]

    note_positions_1_transferred_to_2 = transfer_note_positions(
        wp, note_ann_1, feature_rate
    )

    absolute_errors_at_voice_notes = np.abs(
        note_ann_2 - note_positions_1_transferred_to_2
    )
    errors_at_voice_notes = note_ann_2 - note_positions_1_transferred_to_2

    misalignments = np.zeros(len(tolerances))

    for idx, tolerance in enumerate(tolerances):  # in milliseconds
        misalignments[idx] = np.mean(
            (absolute_errors_at_voice_notes > tolerance / 1000.0)
        )

    mean = np.mean(absolute_errors_at_voice_notes) * 1000.0
    std = np.std(absolute_errors_at_voice_notes) * 1000.0

    return (
        mean,
        std,
        np.array(misalignments),
        errors_at_voice_notes,
        absolute_errors_at_voice_notes,
    )


def run_evaluation(
    wp_dict, ann_1="ref", ann_2="target", norm=None, metric=None, features=FEATURES
):
    stats_dict = dict()
    for song_id in tqdm(wp_dict.keys()):
        wp_chroma_dlnco = wp_dict[song_id]["wp_chroma_dlnco"]

        note_ann_file_1 = NOTE_ANN_DIR / f"ann_{FILENAME_PREFIX}{song_id}_{ann_1}.csv"
        note_ann_file_2 = NOTE_ANN_DIR / f"ann_{FILENAME_PREFIX}{song_id}_{ann_2}.csv"
        mean, std, misalignments, err, abs_err = get_stats(
            wp=wp_chroma_dlnco,
            note_ann_filepath_1=note_ann_file_1.as_posix(),
            note_ann_filepath_2=note_ann_file_2.as_posix(),
        )

        stats_dict[song_id] = dict()
        stats_dict[song_id]["mean"] = mean
        stats_dict[song_id]["std"] = std
        stats_dict[song_id]["misalignments"] = misalignments
        stats_dict[song_id]["errors"] = err
        stats_dict[song_id]["absolute_errors"] = abs_err

    rows = pd.MultiIndex.from_product([stats_dict.keys()], names=["Song ID"])
    column_names = ["Mean(ms)", "Median(ms)", "Std"] + TOLERANCES
    columns = pd.MultiIndex.from_product(
        [[f"{features} / norm={norm}, metric={metric}"], column_names],
        names=["Feature Type", "$\u03C4$ (ms)"],
    )

    # display results
    data = np.zeros((len(stats_dict), len(column_names)))
    for row_idx, song_id in enumerate(stats_dict):
        data[row_idx, 0] = stats_dict[song_id]["mean"]
        data[row_idx, 1] = np.median(stats_dict[song_id]["absolute_errors"]) * 1000.0
        data[row_idx, 2] = stats_dict[song_id]["std"]
        data[row_idx, 3:] = 100 - stats_dict[song_id]["misalignments"] * 100

    df = pd.DataFrame(data, index=rows, columns=columns)
    with pd.option_context("display.float_format", "{:0.2f}".format):
        ipd.display(df)

    return stats_dict
