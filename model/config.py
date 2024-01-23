from pathlib import Path
import numpy as np


DEFAULT_PIECE_ID = 12
CHANNELS = 1
SAMPLE_RATE = 16000
HOP_LENGTH = 640  # 40ms
N_FFT = 2 * HOP_LENGTH
N_MELS = 66
NORM = np.inf
CHUNK_SIZE = 4 * HOP_LENGTH
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH
DTW_WINDOW_SIZE = int(3 * FRAME_RATE)
CRNN_MODEL_PATH = Path("./model/pretrained-model.pt")
FEATURES = ["chroma", "phoneme"]  # chroma, mel, phoneme, mfcc, etc.


# SWD Evaluation Config
TOLERANCES = [200, 300, 500, 750, 1000]
SWD_DATASET_PATH = Path("./data/winterreise_rt")
AUDIO_DIR = SWD_DATASET_PATH / "01_RawData/audio_wav/"
SCORE_DIR = SWD_DATASET_PATH / "01_RawData/score_musicxml/"
LYRICS_DIR = SWD_DATASET_PATH / "01_RawData/lyrics_txt/"
NOTE_ANN_DIR = SWD_DATASET_PATH / "02_Annotations/ann_audio_note/"
WP_ANN_DIR = SWD_DATASET_PATH / "02_Annotations/ann_audio_wp/"
FILENAME_PREFIX = "Schubert_D911-"
