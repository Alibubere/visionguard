import logging
import os
import json
import shutil


def merge_coco_annotations(data_path, output_path, train_img_dir, val_img_dir):
    """
    Merge multiple COCO-style datasets (one per product) into a single
    global train/val COCO dataset, and copy all images into unified
    train/val image directories.

    Args:
        data_path (str): Root directory containing product folders.
        output_path (str): Directory to save merged annotations.
        train_img_dir (str): Directory to store merged train images.
        val_img_dir (str): Directory to store merged val images.
    """

    if not os.path.exists(data_path):
        logging.error(f"Data path does not exist: {data_path}")
        return None

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    # Global storage
    global_train_images = []
    global_train_annotations = []
    global_val_images = []
    global_val_annotations = []
    global_categories = {}

    # Global counters
    global_image_id = 1
    global_annotation_id = 1
    global_category_id = 1

    try:
        for product_folder in os.listdir(data_path):
            product_path = os.path.join(data_path, product_folder)

            if not os.path.isdir(product_path):
                logging.warning(f"Skipping {product_path}, not a directory")
                continue

            product_name = product_folder

            train_json = os.path.join(product_path, "train", "_annotations.coco.json")
            val_json = os.path.join(product_path, "val", "_annotations.coco.json")

            if not os.path.exists(train_json) or not os.path.exists(val_json):
                logging.warning(
                    f"Missing train/val annotation file for {product_name}, skipping."
                )
                continue

            train_image_map = {}
            val_image_map = {}

            # Load train data
            with open(train_json, "r") as f:
                train_data = json.load(f)

            # Load val data
            with open(val_json, "r") as f:
                val_data = json.load(f)

            # ----------------------------------------------------
            # MERGE TRAIN CATEGORIES
            # ----------------------------------------------------
            for category in train_data.get("categories", []):
                category_name = category["name"]

                if category_name not in global_categories:
                    global_categories[category_name] = global_category_id
                    global_category_id += 1

            # ----------------------------------------------------
            # MERGE TRAIN IMAGES (copy into unified train_img_dir)
            # ----------------------------------------------------
            for image in train_data.get("images", []):
                old_image_id = image["id"]
                new_id = global_image_id

                # Assign new global image id
                image["id"] = new_id
                train_image_map[old_image_id] = new_id

                # Build original image path
                original_file_name = image["file_name"]
                src_img_path = os.path.join(product_path, "train", original_file_name)

                if not os.path.exists(src_img_path):
                    logging.error(f"Train image not found: {src_img_path}")
                    continue

                # Create unique new file name
                ext = os.path.splitext(original_file_name)[1]
                new_file_name = f"{product_name}_{new_id:06d}{ext}"
                dst_img_path = os.path.join(train_img_dir, new_file_name)

                try:
                    shutil.copy2(src_img_path, dst_img_path)
                except Exception:
                    logging.exception(f"Failed to copy train image: {src_img_path}")
                    continue

                # Update file_name and optionally store product metadata
                image["file_name"] = new_file_name
                image["product"] = product_name

                global_train_images.append(image)
                global_image_id += 1

            # ----------------------------------------------------
            # MERGE TRAIN ANNOTATIONS
            # ----------------------------------------------------
            local_train_cat_map = {}
            for cat in train_data.get("categories", []):
                local_train_cat_map[cat["id"]] = cat["name"]

            for ann in train_data.get("annotations", []):
                # New global annotation id
                ann["id"] = global_annotation_id
                global_annotation_id += 1

                # Remap image_id
                old_image_id = ann["image_id"]
                if old_image_id not in train_image_map:
                    logging.error(f"Train image_id {old_image_id} not in train_image_map")
                    continue
                ann["image_id"] = train_image_map[old_image_id]

                # Remap category_id via global_categories
                local_id = ann["category_id"]
                category_name = local_train_cat_map[local_id]
                ann["category_id"] = global_categories[category_name]

                global_train_annotations.append(ann)

            # ----------------------------------------------------
            # MERGE VAL CATEGORIES
            # ----------------------------------------------------
            for category in val_data.get("categories", []):
                category_name = category["name"]

                if category_name not in global_categories:
                    global_categories[category_name] = global_category_id
                    global_category_id += 1

            # ----------------------------------------------------
            # MERGE VAL IMAGES (copy into unified val_img_dir)
            # ----------------------------------------------------
            for image in val_data.get("images", []):
                old_image_id = image["id"]
                new_id = global_image_id
                image["id"] = new_id
                val_image_map[old_image_id] = new_id

                original_file_name = image["file_name"]
                src_img_path = os.path.join(product_path, "val", original_file_name)

                if not os.path.exists(src_img_path):
                    logging.error(f"Val image not found: {src_img_path}")
                    continue

                ext = os.path.splitext(original_file_name)[1]
                new_file_name = f"{product_name}_{new_id:06d}{ext}"
                dst_img_path = os.path.join(val_img_dir, new_file_name)

                try:
                    shutil.copy2(src_img_path, dst_img_path)
                except Exception:
                    logging.exception(f"Failed to copy val image: {src_img_path}")
                    continue

                image["file_name"] = new_file_name
                image["product"] = product_name

                global_val_images.append(image)
                global_image_id += 1

            # ----------------------------------------------------
            # MERGE VAL ANNOTATIONS
            # ----------------------------------------------------
            local_val_cat_map = {}
            for cat in val_data.get("categories", []):
                local_val_cat_map[cat["id"]] = cat["name"]

            for ann in val_data.get("annotations", []):
                ann["id"] = global_annotation_id
                global_annotation_id += 1

                old_image_id = ann["image_id"]
                if old_image_id not in val_image_map:
                    logging.error(f"Val image_id {old_image_id} not in val_image_map")
                    continue

                ann["image_id"] = val_image_map[old_image_id]

                local_id = ann["category_id"]
                category_name = local_val_cat_map[local_id]
                ann["category_id"] = global_categories[category_name]

                global_val_annotations.append(ann)

        # ----------------------------------------------------
        # BUILD FINAL CATEGORY LIST
        # ----------------------------------------------------
        final_categories = []
        for name, new_id in global_categories.items():
            final_categories.append({"id": new_id, "name": name})
        final_categories.sort(key=lambda x: x["id"])

        logging.info("Final categories created successfully")

        # Output files
        output_train = os.path.join(output_path, "instances_train_all.json")
        output_val = os.path.join(output_path, "instances_val_all.json")

        # SAVE TRAIN JSON
        train_output = {
            "images": global_train_images,
            "annotations": global_train_annotations,
            "categories": final_categories,
        }
        with open(output_train, "w") as f:
            json.dump(train_output, f, indent=2)

        logging.info(f"Successfully stored train annotations in {output_train}")

        # SAVE VAL JSON
        val_output = {
            "images": global_val_images,
            "annotations": global_val_annotations,
            "categories": final_categories,
        }
        with open(output_val, "w") as f:
            json.dump(val_output, f, indent=2)

        logging.info(f"Successfully stored val annotations in {output_val}")

    except Exception as e:
        logging.exception(f"Failed to merge datasets: {e}")
