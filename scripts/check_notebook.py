"""Check notebook preprocessing and model execution with temporary audio fixtures."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

import numpy as np
import soundfile as sf
import tensorflow as tf


def run_cell(cells, identifier, namespace):
    source = "".join(cells[identifier]["source"])
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, identifier, "exec"), namespace)


def check_arrays(namespace, names, shape):
    for name in names:
        array = namespace[name]
        assert array.shape[1:] == shape, (name, array.shape)
        assert np.isfinite(array).all(), name


def check_model(model, inputs, labels, output_path):
    prediction = model(inputs[:1], training=False).numpy()
    assert prediction.shape == (1, 3)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction.sum(axis=1), 1.0, atol=1e-5)
    result = model.train_on_batch(inputs[:1], np.asarray(labels[:1]))
    assert np.isfinite(np.asarray(result)).all(), result
    expected = model(inputs[:1], training=False).numpy()
    model.save(output_path)
    restored = tf.keras.models.load_model(output_path)
    np.testing.assert_allclose(restored(inputs[:1], training=False).numpy(), expected, atol=1e-5)


def main():
    notebook = Path(__file__).resolve().parents[1] / "AnimalCNN.ipynb"
    document = json.loads(notebook.read_text())
    cells = {cell["id"]: cell for cell in document["cells"]}
    for identifier, cell in cells.items():
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if not source.lstrip().startswith("%pip"):
                compile(source, identifier, "exec")

    with tempfile.TemporaryDirectory(prefix="animal-audio-check-") as directory:
        root = Path(directory)
        for split, count in (("train", 10), ("test", 2)):
            for label, frequency in (("cat", 440), ("dog", 660), ("dolphin", 880)):
                folder = root / "data" / split / label
                folder.mkdir(parents=True)
                for index in range(count):
                    rate = (8000, 16000, 44100)[index % 3]
                    times = np.arange(int(1.1 * rate)) / rate
                    audio = 0.3 * np.sin(2 * np.pi * frequency * times)
                    sf.write(folder / f"{index:02d}.wav", audio, rate)

        previous = {key: os.environ.get(key) for key in ("ANIMAL_DATASET_DIR", "ANIMAL_OUTPUT_DIR")}
        os.environ["ANIMAL_DATASET_DIR"] = str(root / "data")
        os.environ["ANIMAL_OUTPUT_DIR"] = str(root / "models")
        namespace = {"__name__": "notebook_check"}
        try:
            for identifier in ("cell-06", "cell-08", "cell-12", "cell-14", "cell-17", "cell-19"):
                run_cell(cells, identifier, namespace)
            check_arrays(namespace, ("X_train_aug", "X_val", "X_test"), (16000, 1))
            for rate in (8000, 16000, 44100):
                file = root / f"signal-{rate}.wav"
                sf.write(file, np.sin(2 * np.pi * 440 * np.arange(rate) / rate), rate)
                audio = namespace["preprocess_audio"](file)
                assert len(audio) == 16000
                peak = np.fft.rfftfreq(len(audio), 1 / 16000)[np.argmax(np.abs(np.fft.rfft(audio)))]
                assert abs(peak - 440) < 2, (rate, peak)
            run_cell(cells, "cell-23", namespace)
            check_model(namespace["model"], namespace["X_train_aug"], namespace["y_train_aug"], root / "waveform.keras")
            print("Waveform preprocessing, frequency preservation, optimisation and save/load: passed")

            run_cell(cells, "cell-36", namespace)
            assert np.isfinite(namespace["normalize_audio"](np.zeros(16000))).all()
            run_cell(cells, "cell-38", namespace)
            check_arrays(namespace, ("X_train", "X_val", "X_test"), (128, 172, 1))
            run_cell(cells, "cell-45", namespace)
            check_model(namespace["model2"], namespace["X_train"], namespace["y_train"], root / "mel.keras")
            print("Mel preprocessing, silent normalisation, optimisation and save/load: passed")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if "plt" in namespace:
                namespace["plt"].close("all")

    print("Notebook checks passed. No classification benchmark was run.")


if __name__ == "__main__":
    main()
