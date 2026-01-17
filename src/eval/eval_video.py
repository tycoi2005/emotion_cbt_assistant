"""Evaluate trained IEMOCAP video emotion model on validation split."""

from pathlib import Path
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import yaml
import os

from config.config import load_config, PROJECT_ROOT
from src.utils.logging_utils import get_logger
from src.data.iemocap_multimodal_dataset import load_iemocap_multimodal_dataset
from src.models.video_iemocap_resnet import build_iemocap_video_model

logger = get_logger("video_eval")


def evaluate_video_model():
    """Main evaluation function for IEMOCAP video model."""
    cfg = load_config()
    logger.info("Loaded configuration")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load raw config for video specific parameters
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    video_cfg_raw = raw_cfg.get("iemocap_video", {})
    image_size = video_cfg_raw.get("image_size", 224)
    backbone = video_cfg_raw.get("backbone", "resnet18")


    logger.info("Loading IEMOCAP multimodal dataset (video only, validation split)...")
    try:
        # Load validation dataset as test set might not have labels
        val_dataset = load_iemocap_multimodal_dataset(
            cfg=cfg,
            modalities=["video"],
            split="val", # Using val split for evaluation
            image_size=image_size,
            is_training=False,  # No augmentation for evaluation
        )
        if val_dataset is None:
            logger.error("Validation dataset for IEMOCAP video not found. Aborting evaluation.")
            return

        # Dynamically determine num_labels and label_names from the dataset
        num_labels = val_dataset.num_labels
        label_names = val_dataset.emotion_labels
        logger.info(f"Evaluating on IEMOCAP video validation set with {len(val_dataset)} samples.")
        logger.info(f"Number of classes: {num_labels}")
        logger.info(f"Emotion labels: {label_names}")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        raise
    
    # Create DataLoader for the validation set
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.text_model.batch_size, # Re-using batch size from text_model config
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


    logger.info("Building video model...")
    model = build_iemocap_video_model(
        num_classes=num_labels, # Use dynamically determined num_labels
        backbone=backbone,
        freeze_backbone=False, # We want to evaluate the full model
        device=device,
    )

    model_path = cfg.paths.models_dir / "video" / "video_iemocap_resnet_best.pt"
    logger.info(f"Attempting to load model weights from: {model_path}")

    if model_path.exists():
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully from fine-tuned checkpoint.")
        except Exception as e:
            logger.warning(f"Failed to load fine-tuned model weights: {e}. Using pretrained ResNet weights instead.")
    else:
        logger.warning(f"Fine-tuned video model weights not found at {model_path}. Using pretrained ResNet weights (ImageNet) only.")
        # Model is already initialized with pretrained ResNet by build_iemocap_video_model if no weights are loaded.

    model.eval()

    all_true_labels = []
    all_pred_labels = []

    # Create emotion_to_id mapping from the dataset's emotion_labels
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(val_dataset.emotion_labels)}
    logger.info(f"Emotion to ID mapping: {emotion_to_id}")

    logger.info("Running evaluation on IEMOCAP video development (validation) set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Video Eval]"):
            video_frames = batch["video"].to(device)
            
            # Convert string labels to integer IDs using the created mapping
            label_strings = batch["emotion"]
            label_ids = [emotion_to_id[s] for s in label_strings]
            labels = torch.tensor(label_ids, dtype=torch.long).to(device)

            loss, logits = model(video_frames, labels)
            preds = logits.argmax(dim=-1)

            all_true_labels.extend(labels.cpu().numpy().tolist())
            all_pred_labels.extend(preds.cpu().numpy().tolist())

    logger.info(f"Evaluation complete. Total samples: {len(all_true_labels)}")

    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)

    logger.info("Computing metrics...")
    
    # Ensure label_names is correctly ordered if necessary.
    # The dataset provides emotion_labels, which are already sorted.
    
    class_report = classification_report(
        all_true_labels,
        all_pred_labels,
        target_names=label_names,
        output_dict=True,
        labels=np.arange(num_labels) # Ensure all labels are included in report
    )

    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)

    overall_accuracy = class_report["accuracy"]
    macro_f1 = class_report["macro avg"]["f1-score"]

    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "video_eval_report.json"

    report_dict = {
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
    }

    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Evaluation report saved to: {report_path}")

    logger.info("\nPer-class metrics:")
    for label_name in label_names:
        if label_name in class_report:
            metrics = class_report[label_name]
            logger.info(
                f"  {label_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}"
            )


if __name__ == "__main__":
    evaluate_video_model()
