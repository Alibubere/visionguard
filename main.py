import logging
import os
import yaml
import torch
import json
from src.work_data.merge_coco import merge_coco_annotations
from src.work_data.dataset import COCOMergedDataset
from src.work_data.dataloader import get_dataloader
from src.model import get_model, get_lr_scheduler, get_optimizer
from src.train import train_one_epoch, evaluate_one_epoch

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

data = config["data"]
train_cfg = config["training"]
model_cfg = config["model"]
ckpt_cfg = config["checkpoint"]


def setup_logging():
    """
    Set Up logging for the Entire Pipeline
    """

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info("Logging successfully Initialized")


def load_json_file(file_path):

    try:

        if not os.path.exists(file_path):
            raise FileExistsError(f"File not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        return data

    except FileNotFoundError as e:
        logging.error(str(e))
        return None
    except json.JSONDecodeError:
        logging.exception(f"Invalid JSON in: {file_path}")
        return None
    except Exception:
        logging.exception(f"Unexpected error loading: {file_path}")
        return None


def main():
    setup_logging()

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data Config
        data_path = data["data_dir"]
        output_path = data["output_dir"]
        train_json_path = data["json_train_path"]
        val_json_path = data["json_val_path"]
        train_img_dir = data["train_img_dir"]
        val_img_dir = data["val_img_dir"]

        # Train Config
        batch_size = train_cfg["batch_size"]
        lr = train_cfg["lr"]
        weight_decay = train_cfg["weight_decay"]
        num_epochs = train_cfg["num_epochs"]

        # Model Config
        num_classes = model_cfg["num_classes"]

        # Checkpoint Config
        checkpoint_dir = ckpt_cfg["dir"]
        file_name = ckpt_cfg["filename"]
        full_path = os.path.join(checkpoint_dir,file_name)

        train_json = load_json_file(train_json_path)
        val_json = load_json_file(val_json_path)

        if train_json is None or val_json is None:
            logging.error("Train or val JSON is None. Exiting.")
            return

        train_images = train_json.get("images", [])
        train_annotations = train_json.get("annotations", [])

        val_images = val_json.get("images", [])
        val_annotations = val_json.get("annotations", [])

        logging.info("Merging COCO annotations...")
        merge_coco_annotations(
            data_path=data_path,
            output_path=output_path,
            train_img_dir=train_img_dir,
            val_img_dir=val_img_dir,
        )
        logging.info("Merge completed.")

        train_dataset = COCOMergedDataset(
            images_list=train_images,
            annotations_list=train_annotations,
            img_dir=train_img_dir,
            transforms=None,
        )
        val_dataset = COCOMergedDataset(
            images_list=val_images,
            annotations_list=val_annotations,
            img_dir=val_img_dir,
            transforms=None,
        )

        train_loader = get_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = get_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        model = get_model(num_classes=num_classes)
        optimizer = get_optimizer(model=model, lr=lr, weight_decay=weight_decay)
        scheduler = get_lr_scheduler(optimizer=optimizer)

        for epoch in range(1,num_epochs+1):

            train_avg_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                scheduler=scheduler,
            )

            val_avg_loss = evaluate_one_epoch(
                model=model, dataloader=val_loader, epoch=epoch, device=device
            )
            logging.info(
                f"Epoch: [{epoch}/{num_epochs}]"
                f"Train_loss: {train_avg_loss:.4f} | Val loss: {val_avg_loss:.4f}"
            )

    except Exception:
        logging.exception("Unexpected error in main()")


if __name__ == "__main__":
    main()
