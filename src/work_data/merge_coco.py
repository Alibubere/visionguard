import logging
import os
import json


def merge_coco_annotations(data_path, output_path):

    if not os.path.exists(data_path):
        logging.error("Data path does not exist")
        return None

    os.makedirs(output_path, exist_ok=True)

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

            train_image_map = {}
            val_image_map = {}

            # Load train data
            with open(train_json, "r") as f:
                train_data = json.load(f)

            # Load val data
            with open(val_json, "r") as f:
                val_data = json.load(f)

            # PLACEHOLDER: MERGE TRAIN CATEGORIES

            for category in train_data["categories"]:
                category_name = category["name"]

                if category_name not in global_categories:
                    global_categories[category_name] = global_category_id
                    global_category_id += 1

            # PLACEHOLDER: MERGE TRAIN IMAGES

            for image in train_data["images"]:
                old_image_id = image["id"]
                new_id = global_image_id
                image["id"] = new_id
                train_image_map[old_image_id] = new_id

                image["product"] = product_name
                global_train_images.append(image)

                global_image_id += 1

            # PLACEHOLDER: MERGE TRAIN ANNOTATIONS

            local_train_cat_map = {}
            for cat in train_data["categories"]:
                local_train_cat_map[cat["id"]] = cat["name"]

            for ann in train_data["annotations"]:
                old_annot_id = ann["id"]
                ann["id"] = global_annotation_id
                global_annotation_id += 1

                old_image_id = ann["image_id"]
                if old_image_id not in train_image_map:
                    logging.error(f"{old_image_id} not in train image map")
                    continue
                ann["image_id"] = train_image_map[old_image_id]

                local_id = ann["category_id"]
                category_name = local_train_cat_map[local_id]
                ann["category_id"] = global_categories[category_name]
                global_train_annotations.append(ann)

            # PLACEHOLDER: MERGE VAL CATEGORIES

            for category in val_data["categories"]:
                category_name = category["name"]

                if category_name not in global_categories:
                    global_categories[category_name] = global_category_id
                    global_category_id += 1

            # PLACEHOLDER: MERGE VAL IMAGES

            for image in val_data["images"]:
                old_image_id = image["id"]
                new_id = global_image_id
                image["id"] = new_id
                val_image_map[old_image_id] = new_id

                image["product"] = product_name
                global_val_images.append(image)
                global_image_id += 1

            # PLACEHOLDER: MERGE VAL ANNOTATIONS
            local_val_cat_map = {}
            for cat in val_data["categories"]:
                local_val_cat_map[cat["id"]] = cat["name"]

            for ann in val_data["annotations"]:
                old_annot_id = ann["id"]
                ann["id"] = global_annotation_id
                global_annotation_id += 1

                old_image_id = ann["image_id"]
                if old_image_id not in val_image_map:
                    logging.error(f"{old_image_id} Not in val image map")
                    continue

                ann["image_id"] = val_image_map[old_image_id]

                local_id = ann["category_id"]
                category_name = local_val_cat_map[local_id]
                ann["category_id"] = global_categories[category_name]
                global_val_annotations.append(ann)

        # PLACEHOLDER: BUILD FINAL CATEGORY LIST
        final_categories = []

        for name, new_id in global_categories.items():
            final_categories.append({"id": new_id, "name": name})
        final_categories.sort(key=lambda x: x["id"])

        logging.info("Final categories created successfully")

        # Output files
        output_train = os.path.join(output_path, "instances_train_all.json")
        output_val = os.path.join(output_path, "instances_val_all.json")

        # PLACEHOLDER: SAVE TRAIN JSON
        train_output = {
            "images": global_train_images,
            "annotations": global_train_annotations,
            "categories": final_categories,
        }
        with open(output_train, "w") as f:
            json.dump(train_output, f, indent=2)

        logging.info("Successfully stored train cleaned data")

        # ----------------------------------------------------
        # PLACEHOLDER: SAVE VAL JSON
        # ----------------------------------------------------
        val_output = {
            "images": global_val_images,
            "annotations": global_val_annotations,
            "categories": final_categories,
        }
        with open(output_val, "w") as f:
            json.dump(val_output, f, indent=2)

        logging.info("Successfully stored val cleaned data")

    except Exception as e:
        logging.exception(f"Failed to merge datasets: {e}")
