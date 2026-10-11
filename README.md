# Animal Sound Classification

I compared two neural-network models for recognising cats, dogs and dolphins from audio recordings: a CNN with a bidirectional LSTM trained on waveforms, and a CNN trained on Mel spectrograms. The project covers audio exploration, augmentation, training curves, confusion matrices and classification metrics.

## Original results

The [original notebook](notebooks/AnimalCNN-original.ipynb) preserves the code, explanations and saved outputs of the coursework submission.

| Model | Test accuracy | Correct predictions | Test recordings |
| --- | ---: | ---: | ---: |
| Waveform CNN/BiLSTM | 93.88% | 92 | 98 |
| Mel-spectrogram CNN | 98.98% | 97 | 98 |

The test set contains 39 cat, 28 dog and 31 dolphin recordings. These are historical results; they have not been reproduced with the maintained notebook. The recovered dataset contains an identical dog recording in training and test, so the original scores should not be treated as a benchmark free of data leakage. Different preprocessing and input durations mean the comparison does not isolate the effect of network architecture.

## Maintained notebook

[AnimalCNN.ipynb](AnimalCNN.ipynb) contains a version prepared for execution locally or in Google Colab, with outputs cleared. It retains both network architectures and makes the dataset and output paths configurable.

| Pipeline | Input | Network |
| --- | --- | --- |
| Waveform | 16,000 samples at 16 kHz: one second, cropped or padded | Conv1D, bidirectional LSTM, global average/max pooling |
| Mel spectrogram | Four seconds at 22,050 Hz, converted to a 128 × 172 spectrogram | Conv2D blocks, global average pooling, dense layers |

Before splitting, the maintained notebook compares decoded audio and excludes training recordings identical to a test recording. The recovered dataset contains one such dog recording; the test set is kept unchanged.

Both pipelines split training and validation data before augmentation, use training statistics for normalisation and evaluate on the separate test directory. The validation fraction is 20% and the random seed is 42.

The maintained version fixes waveform resampling, protects normalisation of silent signals and gives two transforms names that describe their implementation: pre-emphasis with harmonic separation, and a gain ramp over time. It also corrects the frequency axis in the exploratory plots. These changes, sorted file loading and a common 50-epoch limit affect reproduction of the original run.

The Mel pipeline retains the original fivefold multiplier for the dog class weight. This is an experimental choice, not a validated general setting. Seeds improve repeatability within an environment; they do not guarantee identical results across TensorFlow versions or hardware.

## Dataset

Use the following directory structure:

```text
cats_dogs_dolphins/
├── train/
│   ├── cat/
│   ├── dog/
│   └── dolphin/
└── test/
    ├── cat/
    ├── dog/
    └── dolphin/
```

The course dataset and trained weights are not distributed here. Provide WAV files in all six directories. Paths are checked before preprocessing; each class also needs enough training recordings for a stratified validation split.

The recovered archive contains 426 mono WAV files. After excluding the duplicate from training, the maintained notebook uses 327 recordings for training/validation and all 98 test recordings:

| Class | Training/validation | Test |
| --- | ---: | ---: |
| Cat | 125 | 39 |
| Dog | 84 | 28 |
| Dolphin | 118 | 31 |

The duplicate is `train/dog/dog_barking_14.wav`, identical after decoding to `test/dog/dog_barking_15.wav`. Exact-audio matching does not detect every possible overlap, such as different excerpts from the same source recording. Dataset provenance and redistribution terms still need to be recovered.

## Run locally

The dependencies were checked with Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyterlab
export ANIMAL_DATASET_DIR="/absolute/path/to/cats_dogs_dolphins"
jupyter lab AnimalCNN.ipynb
```

Run the notebook from top to bottom. The first cell installs the same pinned runtime dependencies as `requirements.txt`; it can be skipped if they are already installed. Without `ANIMAL_DATASET_DIR`, the local default is `data/cats_dogs_dolphins`.

Edit the configuration cell to change `MAX_EPOCHS`, the validation fraction or the dog class weight multiplier. Outputs go to `outputs/`, or to `ANIMAL_OUTPUT_DIR` when set. Training writes `audio_waveform_model.keras`, `spectrogram_model.keras` and the Mel checkpoint `best_model.keras`.

For a complete run without the interactive interface:

```bash
python scripts/run_notebook.py --dataset /absolute/path/to/cats_dogs_dolphins --output outputs/run
```

The runner saves an executed notebook, figures, epoch logs and recovery checkpoints. Use `--epochs 1` for a brief execution check; its metrics are not a converged benchmark. Full training can take several hours on a laptop CPU.

## Run in Colab

Upload the maintained notebook to Colab, run the dependency cell and restart the session if the environment requests it. The configuration cell mounts Drive when the Colab module is available. The default dataset path is `/content/drive/MyDrive/cats_dogs_dolphins`; edit it or set `ANIMAL_DATASET_DIR` if your folder is elsewhere. A GPU is useful for training the two models.

## Verification

Run the isolated check without the course dataset:

```bash
python scripts/check_notebook.py
```

The check compiles notebook code, generates temporary WAVs at 8, 16 and 44.1 kHz, checks exclusion of a deliberately duplicated recording, runs both preprocessing and augmentation pipelines, checks silent-signal handling and verifies both model outputs. It also performs one optimisation step and a save/load round trip for each model. It does not estimate classification performance.

These checks passed on CPU with the pinned environment. A one-epoch execution check also completed both pipelines on the recovered real dataset, including test evaluation, model save/load, classification reports and 13 figures. The waveform model used its saved first-epoch checkpoint; the Mel model trained for one epoch in the same execution.

| One-epoch check | Correct predictions | Test recordings | Accuracy |
| --- | ---: | ---: | ---: |
| Waveform CNN/BiLSTM | 40 | 98 | 40.82% |
| Mel-spectrogram CNN | 58 | 98 | 59.18% |

These scores only document the execution check; one epoch is not a converged benchmark. Full training remains to be rerun. Machine-readable summaries are in [validation/](validation/). The original results above belong exclusively to the archived notebook.

## Context

Individual academic project completed during my Master's studies in Computer Engineering at the University of Rome Tor Vergata.
