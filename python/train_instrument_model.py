from torch import multiprocessing
from torch.utils.data import DataLoader
from model import build_model
from early_stopping import EarlyStopping
import json
from hcqt_preprocessing import create_datasets
import traceback
from data_processing import *
from config import *
from print_functions import *

def calculate_metrics(preds, targets, thresholds=None):
    if thresholds is None:
        thresholds = np.full(preds.shape[1], 0.5)

    pred_binary = np.zeros_like(preds)
    for i in range(preds.shape[1]):
        pred_binary[:, i, :] = (preds[:, i, :] > thresholds[i]).astype(int)

    target_binary = targets.astype(int)

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for i in range(preds.shape[1]):
        pred_flat = pred_binary[:, i, :].flatten()
        target_flat = target_binary[:, i, :].flatten()

        tp = np.sum((pred_flat == 1) & (target_flat == 1))
        fp = np.sum((pred_flat == 1) & (target_flat == 0))
        fn = np.sum((pred_flat == 0) & (target_flat == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    frame_accuracy = np.mean(np.all(pred_binary == target_binary, axis=(1, 2)))

    hamming = np.mean(pred_binary != target_binary)

    return {
        'per_class': {
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores
        },
        'macro': {
            'precision': float(np.mean(precision_scores)),
            'recall': float(np.mean(recall_scores)),
            'f1': float(np.mean(f1_scores))
        },
        'frame_accuracy': float(frame_accuracy),
        'hamming_loss': float(hamming)
    }


def evaluate_model(model, data_loader, criterion, device, weights_info):
    model.eval()

    total_loss = 0
    all_predictions = []
    all_targets = []
    all_frame_times = []
    all_filenames = []

    with torch.no_grad():
        for batch_data in data_loader:
            data, target, filenames, frame_times = batch_data

            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            predictions = torch.sigmoid(output).cpu().numpy()
            targets = target.cpu().numpy()

            all_predictions.append(predictions)
            all_targets.append(targets)
            all_frame_times.extend(frame_times.tolist() if torch.is_tensor(frame_times) else frame_times)
            all_filenames.extend(filenames)

    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    activity_metrics = calculate_metrics(
        all_predictions, all_targets, weights_info['thresholds']
    )

    avg_loss = total_loss / len(data_loader)

    return avg_loss, activity_metrics, all_predictions, all_targets

def save_epoch_stats(stats, log_file):
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                all_stats = json.load(f)
        else:
            all_stats = {'epochs': []}

        all_stats['epochs'].append(stats)

        with open(log_file, 'w') as f:
            json.dump(all_stats, f, indent=2)

    except Exception as e:
        print(f"Could not save epoch stats: {e}")

def train_model(model, train_loader, val_loader, test_loader, device, log_file, save_path):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=OPTIM_WEIGHT_DECAY,
        eps=OPTIM_EPS
    )

    print("\nAnalyzing label distribution for weighting:")
    dataset_hash = generate_dataset_hash(train_loader.dataset, val_loader.dataset, test_loader.dataset)

    training_stats, train_target_stats = compute_dataset_statistics(
        train_loader, dataset_hash, device, "Training"
    )

    print("\nAnalyzing validation set statistics:")
    val_stats, val_target_stats = compute_dataset_statistics(
        val_loader, dataset_hash, device, "Validation"
    )

    print("\nAnalyzing test set statistics:")
    test_stats, test_target_stats = compute_dataset_statistics(
        test_loader, dataset_hash, device,  "Testing"
    )

    weights_info = calculate_class_weights(training_stats, device)
    pos_weights = weights_info['pos_weights'].unsqueeze(1).expand(-1, training_stats['time_steps'])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    datasets_info = {
        'Training': (training_stats, train_target_stats),
        'Validation': (val_stats, val_target_stats),
        'Test': (test_stats, test_target_stats)
    }

    for dataset_name, (temp_stats, tgt_stats) in datasets_info.items():
        print(f"\n{dataset_name} Dataset Statistics:")
        print(f"Total samples: {tgt_stats['total_samples']}")

        print(f"Activity prevalence:")
        for i, instrument in enumerate(INSTRUMENTS_LABELS):
            temp_prevalence = tgt_stats['temporal_mean'][i]
            temp_count = tgt_stats['temporal_sum'][i]
            print(f"  {instrument}: {temp_count:.0f} active steps ({temp_prevalence:.6f} prevalence)")

        print(f"\nUsing Weighted BCELoss:")
        print(f"pos_weights: {weights_info['pos_weights'].cpu().numpy()}")
        print(f"Thresholds: {weights_info['thresholds']}")

    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-7
    )

    early_stopping = EarlyStopping(patience=ES_PATIENCE, delta=ES_DELTA, save_path=save_path, metric_name=ES_METRIC)

    print("\nStarting model training")

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 80)

        model.train()
        epoch_total_loss = 0
        batch_count = 0

        for batch_idx, batch_data in enumerate(train_loader):
            data, target, _, _ = batch_data
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            total_loss = loss

            total_loss.backward()

            if batch_idx % 10 == 0:
                print(f"\nBatch {batch_idx + 1} Analysis:")
                grad_norm, layer_info = gradient_analysis(model)
                print(f"Batch {batch_idx + 1}/{len(train_loader)}")
                print(f"Total Loss: {total_loss.item():.6f}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            batch_count += 1

        avg_total_loss = epoch_total_loss / batch_count

        val_loss, val_metrics, val_predictions, val_targets = evaluate_model(
            model, val_loader, criterion, device, weights_info
        )

        epoch_time = time.time() - epoch_start_time
        print(f"Validation Metrics:")
        print(f"Validation Loss: {val_loss:.6f}")
        print_results(val_metrics, INSTRUMENTS_LABELS)

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Training Loss: {avg_total_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Epoch Time: {epoch_time:.2f}s")

        if epoch < WARMUP_EPOCHS:
            lr_scale = (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE * lr_scale
        else:
            main_scheduler.step()

        early_stopping(val_loss, model, optimizer, None, epoch, weights_info['thresholds'])

        epoch_stats = {
            'epoch': epoch + 1,
            'timestamp': time.time(),
            'epoch_time': epoch_time,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'train_loss': avg_total_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'weights_info': {
                'temporal_thresholds': weights_info['thresholds'].tolist()
            },
            'dataset_hash': dataset_hash
        }

        save_epoch_stats(epoch_stats, log_file)

        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        if epoch % 5 == 0:
            if not check_model_health(model, val_loader, device, True):
                print("Model health deteriorated! Stopping training.")
                break

    print("\nTraining completed!")

    print("Loading best model for final evaluation...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, _, val_preds_raw, val_targets_raw = evaluate_model(
        model, val_loader, criterion, device, weights_info
    )

    y_pred_reshaped = np.transpose(val_preds_raw, (1, 0, 2))
    y_true_reshaped = np.transpose(val_targets_raw, (1, 0, 2))

    num_instruments = y_pred_reshaped.shape[0]
    y_pred_reshaped = y_pred_reshaped.reshape(num_instruments, -1)
    y_true_reshaped = y_true_reshaped.reshape(num_instruments, -1)

    val_preds_flat = np.transpose(y_pred_reshaped)
    val_targets_flat = np.transpose(y_true_reshaped)

    val_targets_flat_binary = (val_targets_flat > 0.5).astype(int)

    print("\nFinding optimal thresholds on raw validation data...")
    optimal_thresholds = find_optimal_thresholds(val_targets_flat_binary, val_preds_flat)
    weights_info['thresholds'] = optimal_thresholds

    try:
        if 'weights_info' not in checkpoint:
            checkpoint['weights_info'] = {}

        checkpoint['weights_info']['temporal_thresholds'] = optimal_thresholds.tolist() if hasattr(optimal_thresholds, 'tolist') else optimal_thresholds
        torch.save(checkpoint, save_path)

        print("\nSuccessfully updated the model checkpoint with new optimal thresholds!")
        print(f"Saved to: {save_path}")

    except Exception as e:
        print(f"\nError: Could not update checkpoint with thresholds: {e}")

    print("-" * 80)
    print("Final Test Set Evaluation (using optimized thresholds):")
    print("-" * 80)
    test_loss, test_metrics, test_predictions, test_targets = evaluate_model(
        model, test_loader, criterion, device, weights_info
    )

    final_stats = {
        'training_completed': True,
        'final_epoch': epoch + 1,
        'early_stopped': early_stopping.early_stop,
        'best_val_loss': early_stopping.best_loss,
        'final_test_losses': test_loss,
        'final_test_metrics': test_metrics,
        'total_parameters': model.count_parameters(),
        'final_weights_info': {
            'temporal_thresholds': weights_info['thresholds'].tolist()
        },
        'dataset_hash': dataset_hash
    }

    save_epoch_stats(final_stats, log_file)
    return model

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        os.makedirs('./models', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./dataset_metadata', exist_ok=True)

        SAVE_PATH = f'./models/instrument_model_{timestamp}.pth'
        LOG_FILE = f'./logs/training_log_{timestamp}.json'

        print("Starting Instrument Detection Training")
        print("-" * 80)
        print(f"Training session: {timestamp}")
        print(f"Model will be saved as: {SAVE_PATH}")
        print(f"Training log will be saved as: {LOG_FILE}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()

        print("\nLoading datasets:")
        try:
            train_dataset, val_dataset, test_dataset = create_datasets(
                label_smoothing=LABEL_SMOOTHING,
            )

            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            print(f"Testing samples: {len(test_dataset)}")

        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, persistent_workers=PERSISTENT_WORKERS,
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=True,
            prefetch_factor=PREFETCH_FACTOR
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, persistent_workers=PERSISTENT_WORKERS,
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=True,
            prefetch_factor=PREFETCH_FACTOR
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, persistent_workers=PERSISTENT_WORKERS,
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=True,
            prefetch_factor=PREFETCH_FACTOR
        )

        print(f"\nInitializing model.")
        model = build_model().to(device)

        trained_model = train_model(model, train_loader, val_loader, test_loader, device, LOG_FILE, SAVE_PATH)

        print(f"\nTraining logs saved to: {LOG_FILE}")
        print(f"Best model saved to: {SAVE_PATH}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    except Exception as e:
        print(f"\nError during training: {e}")
        traceback.print_exc()