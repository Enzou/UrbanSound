import os
from pathlib import Path
import pandas as pd
import librosa


DATA_DIR = Path('./data')


def main():
    data_dir = Path('../data/')
    load_data(data_dir)
    pass


def load_data(src_path):
    samples = []
    for file_name in os.listdir(src_path):
        data, sampling_rate = librosa.load(src_path/file_name)
        samples.append((data, sampling_rate))
    return samples


if __name__ == "__main__":
    # models to use:
    # - MLP
    # - RNN
    #   - LSTM
    #   - GRU
    #   - std. RNN
    # - CNNs
    # - Transformer ?
    main()
