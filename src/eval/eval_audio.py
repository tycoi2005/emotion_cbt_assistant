import sys
from pathlib import Path

# Add project root to sys.path automatically
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from datetime import datetime
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config.config import load_config
from src.utils.logging_utils import get_logger
from src.data.daicwoz_audio_dataset import create_daicwoz_datasets
from src.models.audio_cnn_lstm import build_audio_model
from src.models.audio_wav2vec_attention import build_wav2vec_attention_model

logger = get_logger("audio_eval")


def evaluate_audio_model():
    """Main evaluation function for audio model on DAIC-WOZ."""
    cfg = load_config()
    logger.info("Loaded configuration")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Creating DAIC-WOZ dataloaders...")
    try:
        # Get DAIC config
        import yaml
        from config.config import PROJECT_ROOT
        cfg_path = PROJECT_ROOT / "config" / "config.yaml"
        with open(cfg_path, 'r') as f:
            daic_cfg = yaml.safe_load(f).get("daic_woz", {})

        _, val_dataset, test_dataset = create_daicwoz_datasets(
            cfg=cfg,
            sample_rate=daic_cfg.get("sample_rate", 16000),
            n_mfcc=daic_cfg.get("n_mfcc", 40),
            max_duration=daic_cfg.get("window_size", 10.0),
            window_size=daic_cfg.get("window_size", 10.0),
            hop_size=daic_cfg.get("hop_size", 5.0),
            feature_type=daic_cfg.get("feature_type", "mfcc"),
            augment=False
        )

        eval_set = val_dataset
        if eval_set is None:
            logger.error("No DAIC-WOZ evaluation dataset found. Aborting evaluation.")
            return

        # Create DataLoader for the evaluation set
        eval_loader = torch.utils.data.DataLoader(
            eval_set,
            batch_size=daic_cfg.get("batch_size", 16),
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info(f"Evaluating on {'test' if test_dataset else 'val'} set with {len(eval_set)} samples.")

    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}", exc_info=True)
        raise

    num_classes = val_dataset.num_labels
    num_mfcc = val_dataset.n_mfcc
    logger.info(f"Number of classes from dataset: {num_classes}")
    logger.info(f"Number of MFCCs from dataset: {num_mfcc}")

    feature_type = daic_cfg.get("feature_type", "mfcc")
    logger.info(f"Building audio model ({feature_type})...")

    if feature_type == "raw":
        model = build_wav2vec_attention_model(num_labels=num_classes, device=device)
        model_path = cfg.paths.models_dir / "audio" / "audio_wav2vec2_attention_best.pt"
        # Fallback to general name if not found
        if not model_path.exists():
            model_path = cfg.paths.models_dir / "audio" / "audio_cnn_lstm_best.pt"
    else:
        model = build_audio_model(num_labels=num_classes, num_mfcc=num_mfcc, device=device)
        model_path = cfg.paths.models_dir / "audio" / "audio_cnn_lstm_best.pt"

    logger.info(f"Loading model weights from: {model_path}")

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}", exc_info=True)
        raise

    model.eval()

    all_true_labels = []
    all_pred_labels = []

    logger.info("Running evaluation (limited to 50 batches)...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader, desc="[Audio Eval]")):
            if i >= 50: # Limit for speed
                break
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            loss, logits = model(features, labels)
            preds = logits.argmax(dim=-1)

            all_true_labels.extend(labels.cpu().numpy().tolist())
            all_pred_labels.extend(preds.cpu().numpy().tolist())

    logger.info(f"Evaluation complete. Total samples: {len(all_true_labels)}")

    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)

    logger.info("Computing metrics...")
    label_names = ["non-depressed", "depressed"]

    class_report = classification_report(
        all_true_labels,
        all_pred_labels,
        target_names=label_names,
        output_dict=True,
    )

    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)

    overall_accuracy = class_report["accuracy"]
    macro_f1 = class_report["macro avg"]["f1-score"]

    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")

    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"metrics_daic_woz_{timestamp}.json"

    # Standardize structure for generate_report.py
    standardized_metrics = {
        "daic_woz": {
            "accuracy": overall_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": class_report["weighted avg"]["f1-score"],
            "per_class": {
                name: {
                    "precision": class_report[name]["precision"],
                    "recall": class_report[name]["recall"],
                    "f1": class_report[name]["f1-score"],
                    "support": class_report[name]["support"]
                } for name in label_names if name in class_report
            },
            "confusion_matrix": conf_matrix.tolist()
        }
    }

    with open(report_path, 'w') as f:
        json.dump(standardized_metrics, f, indent=2)

    logger.info(f"Evaluation report saved to: {report_path}")

    logger.info("\nPer-class metrics:")
    for label_name in label_names:
        if label_name in class_report:
            metrics = class_report[label_name]
            logger.info(
                f"  {label_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}"
            )


if __name__ == "__main__":
    evaluate_audio_model()
