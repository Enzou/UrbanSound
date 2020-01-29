import itertools
import os

import librosa
import pandas as pd
from pathlib import Path
from typing import Optional, List, Generator, Tuple, Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.switch_backend('agg')


def plot_confusion_matrix(conv_mat, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        conv_mat = conv_mat.astype('float') / conv_mat.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
    # print('Confusion matrix, without normalization')

    plt.imshow(conv_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = conv_mat.max() / 2.
    for i, j in itertools.product(range(conv_mat.shape[0]), range(conv_mat.shape[1])):
        plt.text(j, i, format(conv_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conv_mat[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show(block=False)
    return plt


def plot_metric(history, metric: str, best_fn: Callable, name: str, out_dir: Optional = None):
    training_values = history.history[metric]
    valid_values = history.history[f"val_{metric}"]
    epochs = len(training_values)

    top_value = best_fn(training_values)
    top_valid_value = best_fn(valid_values)

    plt.plot(training_values)
    plt.plot(valid_values)

    plt.axhline(y=top_value, color='grey', alpha=0.5)
    plt.annotate("{0:.4f}".format(top_value), xy=(0, top_value), bbox=dict(boxstyle="round4", fc="w", alpha=0.5))
    plt.axhline(y=top_valid_value, color='grey', alpha=0.5)
    plt.annotate("{0:.4f}".format(top_valid_value), xy=(int(epochs/5), top_valid_value), bbox=dict(boxstyle="round4", fc="w", alpha=0.5))

    plt.title(f"{name} model {metric}")
    plt.ylabel(metric.title())
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='center right')
    plt.draw()
    if out_dir is not None:
        plt.savefig(out_dir / f"_{metric}.png")
    plt.show(block=False)
    plt.clf()


def load_data(src_dir: Path, labels_src: str, cache_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load all audio files from the given directory.
    """
    #     if cache_file is None:
    #         cache_file = f"_cached_{src_dir.stem}.pkl"
    cache_dir = src_dir.parent/'interim'  # cached files are in interim sibling-folder

    if cache_file is not None and os.path.exists(cache_dir/cache_file):
        print(f"Loading cached data!")
        return pd.read_pickle(cache_dir/cache_file)

    train_info = pd.read_csv(src_dir / labels_src, dtype={'ID': object})
    id_col = 'ID'
    features = []
    for sample_id, data, sample_rate in load_samples(src_dir, list(train_info['ID'])):
        features.append({
            id_col: sample_id,
            'raw': data,
            'sample_rate': sample_rate,
            'duration': librosa.get_duration(y=data, sr=sample_rate)
            # 'mfcc': extract_features(data, sample_rate)
        })

    df = pd.merge(train_info, pd.DataFrame(features))
    df.set_index(id_col)

    if cache_file is not None:
        df.to_pickle(cache_dir/cache_file)
    return df


def load_samples(src_path: Path, sample_names: Optional[List[str]]) -> Generator[Tuple[Any, Any, Any], None, None]:
    files = [f for f in os.listdir(src_path) if Path(f).stem in sample_names]
    for file_name in tqdm(files):
        file_path = src_path / file_name
        data, sample_rate = librosa.load(file_path)
        yield file_path.stem, data, sample_rate


def extract_features(raw: np.array, sample_rate: int, output_ndim: int = 2, max_pad: int = 0, pad_mode: str = 'constant') -> pd.Series:
    """
    Extract MFCC features from the given audio file.
    :params max_pad: if a padding value > 0 is supplied, then all samples are being paddind to have the same dimension.
    :params pad_mode: mode to use for fill the missing values
    """
    mfccs = librosa.feature.mfcc(y=raw, sr=sample_rate, n_mfcc=40)
    if max_pad > 0:
        pad_width = max_pad - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode=pad_mode)
    if output_ndim == 1:
        mfccs = np.mean(mfccs, axis=1)  # use only mean value per MFCC (across all frames)
        return pd.Series({'features': mfccs, 'n_mfccs': mfccs.shape[0], 'n_samples': len(raw)})
    else:
        return pd.Series({'features': mfccs, 'n_mfccs': mfccs.shape[0], 'n_frames': mfccs.shape[1], 'n_samples': len(raw)})

