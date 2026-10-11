"""Execute the maintained notebook on a WAV dataset and save local results."""

import argparse
import base64
import contextlib
import csv
from types import SimpleNamespace
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/run'))
    parser.add_argument('--epochs', type=int, help='Override the notebook epoch limit')
    parser.add_argument('--waveform-checkpoint', type=Path, help='Evaluate a saved waveform checkpoint instead of training it again')
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        parser.error('Dataset directory does not exist')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['ANIMAL_DATASET_DIR'] = str(dataset)
    os.environ['ANIMAL_OUTPUT_DIR'] = str(output)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')

    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    notebook_path = Path(__file__).resolve().parents[1] / 'AnimalCNN.ipynb'
    notebook = json.loads(notebook_path.read_text())
    namespace = {'__name__': '__main__'}
    started = time.time()
    status = {'state': 'running', 'started_at': started, 'completed_cells': [], 'models': {}}
    current_outputs = []
    fit_count = 0
    figure_count = 0
    original_fit = tf.keras.Model.fit
    original_show = plt.show

    def save_status():
        status['elapsed_seconds'] = time.time() - started
        (output / 'execution-status.json').write_text(json.dumps(status, indent=2)+'\n')
        (output / 'AnimalCNN-executed.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')

    def show_figures(*unused, **kwargs):
        nonlocal figure_count
        for number in plt.get_fignums():
            figure = plt.figure(number)
            buffer = io.BytesIO()
            figure.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
            figure_count += 1
            (output / f'figure-{figure_count:02d}.png').write_bytes(buffer.getvalue())
            current_outputs.append({'output_type':'display_data', 'data':{'image/png':base64.b64encode(buffer.getvalue()).decode()}, 'metadata':{}})
        plt.close('all')

    def logged_fit(model, *fit_args, **fit_kwargs):
        nonlocal fit_count
        fit_count += 1
        name = 'waveform' if fit_count == 1 else 'mel'
        if name == 'waveform' and args.waveform_checkpoint:
            csv_path = args.waveform_checkpoint.resolve().with_name('waveform-epochs.csv')
            with csv_path.open() as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                raise ValueError('No waveform epoch history found beside the checkpoint')
            history = {key:[float(row[key]) for row in rows] for key in rows[0] if key != 'epoch'}
            checkpoint_epoch = int(np.argmin(history['val_loss'])) + 1
            status['models'][name] = {'epochs_completed':len(rows), 'history':history, 'loaded_checkpoint':True, 'checkpoint_epoch':checkpoint_epoch}
            return SimpleNamespace(history=history)
        fit_kwargs['verbose'] = 2
        callbacks = list(fit_kwargs.get('callbacks', []))
        callbacks.append(tf.keras.callbacks.CSVLogger(str(output / f'{name}-epochs.csv')))
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(str(output / f'{name}-recovery.keras'), monitor='val_loss', save_best_only=True))
        fit_kwargs['callbacks'] = callbacks
        history = original_fit(model, *fit_args, **fit_kwargs)
        status['models'][name] = {'epochs_completed':len(history.history['loss']), 'history':{key:[float(x) for x in values] for key,values in history.history.items()}}
        save_status()
        return history

    class Tee(io.TextIOBase):
        def __init__(self, target, buffer):
            self.target, self.buffer = target, buffer
        def write(self, text):
            self.buffer.write(text)
            self.target.write(text)
            self.target.flush()
            return len(text)
        def flush(self):
            self.target.flush()

    plt.show = show_figures
    tf.keras.Model.fit = logged_fit
    try:
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            if source.lstrip().startswith('%pip'):
                continue
            print(f'Executing {cell["id"]}', flush=True)
            status['current_cell'] = cell['id']
            current_outputs = []
            if cell['id'] == 'cell-25' and args.waveform_checkpoint:
                namespace['model'] = tf.keras.models.load_model(args.waveform_checkpoint.resolve())
            save_status()
            buffer = io.StringIO()
            cell['execution_count'] = len(status['completed_cells'])+1
            with contextlib.redirect_stdout(Tee(sys.stdout, buffer)):
                exec(compile(source, cell['id'], 'exec'), namespace)
            if cell['id'] == 'cell-06' and args.epochs is not None:
                namespace['MAX_EPOCHS'] = args.epochs
            cell['outputs'] = [{'output_type':'stream','name':'stdout','text':buffer.getvalue()}] + current_outputs
            status['completed_cells'].append(cell['id'])
            if cell['id'] == 'cell-08':
                status['training_counts'] = {key:len(values) for key,values in namespace['data_paths'].items()}
                status['test_counts'] = {key:len(values) for key,values in namespace['test_data_paths'].items()}
                status['excluded_training_duplicates'] = [{key:str(Path(value).relative_to(dataset)) for key,value in pair.items()} for pair in namespace['excluded_training_duplicates']]
                status['max_epochs'] = namespace['MAX_EPOCHS']
            if cell['id'] in ('cell-30','cell-52'):
                name = 'waveform' if cell['id'] == 'cell-30' else 'mel'
                status['models'][name]['test_accuracy'] = float(namespace['test_accuracy'])
                status['models'][name]['test_loss'] = float(namespace['test_loss'])
                prediction = namespace['y_pred_classes']
                status['models'][name]['correct_predictions'] = int(np.sum(prediction == namespace['y_test']))
                status['models'][name]['test_recordings'] = len(namespace['y_test'])
            save_status()
        status['state'] = 'complete'
        status['figure_count'] = figure_count
    except KeyboardInterrupt:
        status['state'] = 'interrupted'
        raise
    except Exception:
        status['state'] = 'failed'
        status['error'] = traceback.format_exc()
        raise
    finally:
        plt.show = original_show
        tf.keras.Model.fit = original_fit
        save_status()
        plt.close('all')
    print('Notebook execution complete.', flush=True)


if __name__ == '__main__':
    main()
