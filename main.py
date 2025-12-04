import logging
import os
import yaml
import torch
import json
from src.work_data.merge_coco import merge_coco_annotations
from src.work_data.dataset import COCOMergedDataset


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

    except Exception:
        print(f"Unexpected error laoding: {file_path}")
        return None


def main():
    setup_logging()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_path = data["data_dir"]
        output_path = data["output_dir"]
        train_json_path = data["json_train_path"]
        val_json_path = data["json_val_path"]
        train_img_dir = data["train_img_dir"]
        val_img_dir = data["val_img_dir"]



        train_json = load_json_file(train_json_path)
        val_json = load_json_file(val_json_path)

        if train_json is None or val_json is None:
            logging.error("Train or val JSON is None. Exiting.")
            return

        train_images = train_json.get("images",[])
        train_annotations = train_json.get("annotation",[])

        val_images = val_json.get("images",[])
        val_annotations = val_json.get("annotation",[])

        logging.info("Merging COCO annotations...")
        merge_coco_annotations(data_path=data_path, output_path=output_path)
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
            transforms=None
        )

    except Exception:
        logging.exception("Unexpected error in main()")


if __name__ == "__main__":
    main()