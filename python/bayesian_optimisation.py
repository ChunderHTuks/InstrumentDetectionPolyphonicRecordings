import torch
import optuna
import json
import time
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from hcqt_preprocessing import create_datasets
from model import build_model_with_params
from early_stopping import EarlyStopping
from train_instrument_model import evaluate_model
from data_processing import calculate_class_weights, generate_dataset_hash, compute_dataset_statistics
from config import *

STUDY_NAME = "instrument-model-vgg-multihead-loss-minimization-v1"
JOURNAL_LOG_FILE = f"{STUDY_NAME}.log"
ARTIFACT_DIR = "./hpo_artifacts"
N_TRIALS = 39

def run_trial_training(trial, params, train_loader, val_loader, device, weights_info):
    model = build_model_with_params(params['model']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['optimizer']['lr'],
        weight_decay=params['optimizer']['weight_decay'],
        eps=OPTIM_EPS
    )

    pos_weights = weights_info['pos_weights'].unsqueeze(1).expand(-1, train_loader.dataset.cqt_frames_per_audio_frame)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-7)

    trial_artifact_dir = os.path.join(ARTIFACT_DIR, f"trial_{trial.number:04d}")
    early_stopping = EarlyStopping(patience=ES_PATIENCE, save_path=os.path.join(trial_artifact_dir, 'best_model.pth'),
                                   metric_name='Loss')

    epoch_history = []
    print(f"Starting Training Loop for Trial {trial.number}")

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        epoch_train_loss = 0

        for data, target, _, _ in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device, weights_info)

        epoch_duration = time.time() - epoch_start_time
        print(
            f"Trial {trial.number} | Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} | Time: {epoch_duration:.2f}s")

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_f1_macro': val_metrics['macro']['f1'],
            'lr': optimizer.param_groups[0]['lr']
        })

        if epoch < WARMUP_EPOCHS:
            lr_scale = min(1.0, (epoch + 1) / WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = params['optimizer']['lr'] * lr_scale
        else:
            scheduler.step()

        early_stopping(val_loss, model, optimizer, None, epoch, weights_info['thresholds'])
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return early_stopping.best_loss, epoch_history


def objective(trial, train_loader, val_loader, device, weights_info):
    trial_num = trial.number
    trial_artifact_dir = os.path.join(ARTIFACT_DIR, f"trial_{trial_num:04d}")
    os.makedirs(trial_artifact_dir, exist_ok=True)

    print(f"\n{'=' * 80}\nStarting Trial {trial_num:04d}\n{'=' * 80}")

    params = {
        'optimizer': {
            'lr': trial.suggest_float("lr", 1e-6, 1e-2, log=True),
            'weight_decay': trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        },
        'model': {
            'channels': [32, 64, 96],
            'dropout_rates': {
                'piano': trial.suggest_float("dropout_piano", 0.1, 0.8),
                'strings': trial.suggest_float("dropout_strings", 0.1, 0.8),
                'woodwinds': trial.suggest_float("dropout_woodwinds", 0.1, 0.8),
                'brass': trial.suggest_float("dropout_brass", 0.1, 0.8),
            }
        }
    }
    with open(os.path.join(trial_artifact_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Trial {trial_num} Parameters:\n{json.dumps(params, indent=2)}")

    try:
        best_val_loss, epoch_history = run_trial_training(trial, params, train_loader, val_loader, device, weights_info)
        pd.DataFrame(epoch_history).to_csv(os.path.join(trial_artifact_dir, 'epoch_history.csv'), index=False)
        result_summary = {
            'best_validation_loss': best_val_loss,
            'status': 'COMPLETE',
            'num_epochs_run': len(epoch_history),
            'completion_time': datetime.now().isoformat()
        }
        with open(os.path.join(trial_artifact_dir, 'result.json'), 'w') as f:
            json.dump(result_summary, f, indent=2)

        print(f"Trial {trial_num:04d} has completed. Best Val Loss: {best_val_loss:.6f}\n")
        return best_val_loss

    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial_num:04d} has been pruned.")
        result_summary = {'status': 'PRUNED', 'completion_time': datetime.now().isoformat()}
        with open(os.path.join(trial_artifact_dir, 'result.json'), 'w') as f:
            json.dump(result_summary, f, indent=2)
        raise

    except Exception as e:
        print(f"Trail {trial_num:04d} has failed.\nError: {e}")
        import traceback
        traceback.print_exc()
        result_summary = {'status': 'FAILED', 'error': str(e), 'completion_time': datetime.now().isoformat()}
        with open(os.path.join(trial_artifact_dir, 'result.json'), 'w') as f:
            json.dump(result_summary, f, indent=2)
        return float('inf')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("Loading datasets for the HPO study.")
    train_dataset, val_dataset, _ = create_datasets(label_smoothing=LABEL_SMOOTHING)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, persistent_workers=PERSISTENT_WORKERS,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=True, prefetch_factor=PREFETCH_FACTOR
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, persistent_workers=PERSISTENT_WORKERS,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=True, prefetch_factor=PREFETCH_FACTOR
    )
    print("Datasets loaded successfully.")

    print("Calculating class weights for loss function.")
    dataset_hash = generate_dataset_hash(train_dataset, val_dataset, val_dataset)
    training_stats, _ = compute_dataset_statistics(train_loader, dataset_hash, device, "Training")
    weights_info = calculate_class_weights(training_stats, device)
    print("Class weights calculated.")

    lock_obj = optuna.storages.JournalFileOpenLock(JOURNAL_LOG_FILE)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(JOURNAL_LOG_FILE, lock_obj=lock_obj)
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )

    print(f"Starting/resuming study '{STUDY_NAME}'. Running for a total of {N_TRIALS} trials.")
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device, weights_info),
        n_trials=N_TRIALS
    )

    best_trial = study.best_trial
    print("\n--- Best Trial Summary ---")
    print(f"  Trial Number: {best_trial.number}")
    print(f"  Value (Best Validation Loss): {best_trial.value:.6f}")
    print("  Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")