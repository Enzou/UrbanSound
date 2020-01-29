import os
from pathlib import Path
from typing import List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
from comet_ml import Experiment


def main():
    st.title("Urban Sound Challenge")
    data_dir = Path('./data/')

    train_info = pd.read_csv(data_dir / "train_short.csv", dtype={'ID': object})
    train_samples = load_samples(data_dir, list(train_info['ID']))

    show_waveplot(train_samples[0])
    # plot_waves("Test", samples[0][0])


def read_sample_info(file_path: str) -> pd.DataFrame:
    pass


# @st.cache
def load_samples(src_path: Path, sample_names: Optional[List[str]]) -> List[Tuple[Any, Any]]:
    samples = []
    files = [f for f in os.listdir(src_path) if Path(f).stem in sample_names]
    for file_name in tqdm(files):
        data, sampling_rate = librosa.load(src_path / file_name)
        samples.append((data, sampling_rate))
    return samples


# def plot_waves(sound_names, raw_sounds):
#     i = 1
#     # fig = plt.figure(figsize=(25,60), dpi = 900)
#     fig = plt.figure(figsize=(25, 60))
#     for n, f in zip(sound_names, raw_sounds):
#         plt.subplot(10, 1, i)
#         librosa.display.waveplot(np.array(f), sr=22050)
#         plt.title(n.title())
#         i += 1
#     plt.suptitle("Figure 1: Waveplot", x=0.5, y=0.915, fontsize=18)
#     plt.show()


def show_waveplot(sample) -> None:
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(sample[0], sr=sample[1])
    st.pyplot()


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
