"""Full audio training pipeline for CNN+LSTM emotion classifier."""

from pathlib import Path
import sys

# Add project root to sys.path automatically
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm

from config.config import load_config
from src.utils.seed_utils import set_global_seed
from src.utils.logging_utils import get_logger

from src.data.daicwoz_audio_dataset import create_daicwoz_datasets
from src.models.audio_cnn_lstm import build_audio_model
from src.models.audio_wav2vec_attention import build_wav2vec_attention_model


def compute_class_weights(labels, num_classes):
    """
    Compute inverse-frequency class weights.

    Args:
        labels: list of int labels
        num_classes: total number of classes

    Returns:
        torch.tensor of shape (num_classes,)
    """
    # Count how many times each class appears
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        if 0 <= label < num_classes:
            counts[label] += 1.0

    # For class c, weight = 1.0 / max(count_c, 1)
    weights = 1.0 / torch.clamp(counts, min=1.0)

    return weights


def create_audio_dataloaders(cfg):
    """
    Create audio dataloaders for training, validation, and test.

    Args:
        cfg: Project configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
        Missing loaders are returned as None.
    """
    logger = get_logger("audio_data")

    # Get DAIC-WOZ specific config
    raw_cfg = load_config()
    # The load_config returns a ProjectConfig, we need to access the raw dictionary if needed or use the updated ProjectConfig if I updated it
    # Actually, I'll just use the values from the raw config for now to be safe
    import yaml
    from config.config import PROJECT_ROOT
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, 'r') as f:
        daic_cfg = yaml.safe_load(f).get("daic_woz", {})

    train_set, val_set, test_set = create_daicwoz_datasets(
        cfg=cfg,
        sample_rate=daic_cfg.get("sample_rate", 16000),
        n_mfcc=daic_cfg.get("n_mfcc", 40),
        max_duration=daic_cfg.get("window_size", 10.0),
        window_size=daic_cfg.get("window_size", 10.0),
        hop_size=daic_cfg.get("hop_size", 5.0),
        feature_type=daic_cfg.get("feature_type", "mfcc")
    )

    # Check if all three are None
    if train_set is None and val_set is None and test_set is None:
        logger.warning("No DAIC-WOZ audio datasets found. Skipping audio training.")
        return None, None, None

    # Check if train_set is None but some other set exists
    if train_set is None:
        logger.warning("Train DAIC-WOZ split not found. Cannot train audio model.")
        return None, None, None

    # Get list of labels
    labels = [s["label"] for s in train_set.samples]
    num_classes = len(set(labels))

    # Compute class weights
    class_weights = compute_class_weights(labels, num_classes)

    # Compute sample weights
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Create train_loader
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.text_model.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Create val_loader
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.text_model.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    # Create test_loader
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=cfg.text_model.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, logger):
    """
    Train model for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        logger: Logger instance

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Audio Train]")
    for batch in progress_bar:
        # Move tensors to device
        features = batch["features"].to(device)  # (batch, n_mfcc, time)
        labels = batch["label"].to(device)

        # Forward pass
        loss, logits = model(features, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track loss (weighted by batch size)
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def evaluate_epoch(model, data_loader, device):
    """
    Evaluate average loss on val/test.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to evaluate on

    Returns:
        Average validation loss as float, or None if data_loader is None
    """
    if data_loader is None:
        return None

    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="[Val]")
        for batch in progress_bar:
            # Move tensors to device
            features = batch["features"].to(device)  # (batch, n_mfcc, time)
            labels = batch["label"].to(device)

            # Forward pass
            loss, logits = model(features, labels)

            # Track loss
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def train_audio_model():
    """
    Main training function for audio emotion classification.
    """
    cfg = load_config()
    logger = get_logger("audio_training")
    set_global_seed()

    # Create dataloaders
    logger.info("Creating audio dataloaders...")
    train_loader, val_loader, test_loader = create_audio_dataloaders(cfg)

    if train_loader is None:
        logger.warning("Audio dataloaders not available. Skipping audio training.")
        return

    # Infer model parameters
    num_classes = len(set([s["label"] for s in train_loader.dataset.samples]))
    num_mfcc = train_loader.dataset.n_mfcc

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Number of MFCC coefficients: {num_mfcc}")

    # Build model
    logger.info("Building audio model...")
    # Get DAIC config again for consistency
    import yaml
    from config.config import PROJECT_ROOT
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, 'r') as f:
        daic_cfg = yaml.safe_load(f).get("daic_woz", {})

    feature_type = daic_cfg.get("feature_type", "mfcc")

    try:
        if feature_type == "raw":
            model = build_wav2vec_attention_model(
                num_labels=num_classes, device=cfg.device.device
            )
        else:
            model = build_audio_model(
                num_labels=num_classes, num_mfcc=num_mfcc, device=cfg.device.device
            )
        logger.info(f"Model built ({feature_type}) and moved to {cfg.device.device}")
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        raise

    # Create optimizer
    learning_rate = float(daic_cfg.get("learning_rate", 1e-4))
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    logger.info(f"Optimizer: AdamW with lr={learning_rate}")

    # Compute total steps
    num_epochs = int(daic_cfg.get("num_epochs", 30))
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    # Create LambdaLR scheduler
    def lr_lambda(step: int) -> float:
        """Learning rate schedule with warmup and linear decay."""
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)
    logger.info(f"LR scheduler: Linear warmup ({warmup_steps} steps) + decay")

    # Training setup
    best_val_loss = float("inf")
    model_name = "audio_wav2vec2_attention_best.pt" if feature_type == "raw" else "audio_cnn_lstm_best.pt"
    best_model_path = cfg.paths.models_dir / "audio" / model_name
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Best model will be saved to: {best_model_path}")

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{cfg.text_model.num_epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.device.device,
            epoch=epoch,
            logger=logger,
        )
        logger.info(f"Train loss: {train_loss:.4f}")

        # Evaluate
        if val_loader is not None:
            val_loss = evaluate_epoch(model=model, data_loader=val_loader, device=cfg.device.device)
            logger.info(f"Val loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                try:
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"[SAVED] New best audio model to {best_model_path}")
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")

    # Final logging
    logger.info(f"\n{'='*50}")
    logger.info("Audio training complete.")
    if val_loader is not None:
        logger.info(f"Best val loss: {best_val_loss:.4f}")
    else:
        logger.info("Best val loss: N/A")


if __name__ == "__main__":
    train_audio_model()
