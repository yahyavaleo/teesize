import os
import json
import shutil

from tqdm import tqdm
from PIL import Image

CATEGORY_NAME = "short sleeve top"
SOURCE_DIR = "deepfashion2"
OUT_DIR = "shirts"
MIN_SIZE = 224
MAX_SIZE = 960
MAX_WIDTH = 640


def has_category(annotations_file):
    with open(annotations_file, "r") as f:
        data = json.load(f)
        for item in data.values():
            if isinstance(item, dict) and item.get("category_name") == CATEGORY_NAME:
                return True
    return False


def extract_fields(data):
    fields = {}
    for item in data.values():
        if isinstance(item, dict) and item.get("category_name") == CATEGORY_NAME:
            fields["bbox"] = item["bounding_box"]
            fields["landmarks"] = item["landmarks"]
            fields["scale"] = item["scale"]
            fields["occlusion"] = item["occlusion"]
            fields["zoom"] = item["zoom_in"]
            fields["viewpoint"] = item["viewpoint"]
            break
    return fields


def crop_image(image_path, bbox, output_path):
    try:
        with Image.open(image_path) as img:
            cropped_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            cropped_img.save(output_path)
    except Exception as e:
        img = Image.new("RGB", (100, 100), color="white")
        img.save(output_path)


def stage1(subdir):
    print(f"Stage 1: Detecting {CATEGORY_NAME}")
    print("-----------------------------------")

    output = []
    annotations_dir = os.path.join(SOURCE_DIR, subdir, "annotations")

    for filename in tqdm(os.listdir(annotations_dir)):
        if filename.endswith(".json"):
            annotations_file = os.path.join(annotations_dir, filename)
            if has_category(annotations_file):
                index = os.path.splitext(filename)[0]
                output.append(index)

    output.sort()

    os.makedirs("temp", exist_ok=True)
    savepath = os.path.join("temp", "stage1.txt")
    with open(savepath, "w") as f:
        for index in output:
            f.write(index + "\n")

    print()


def stage2(subdir):
    print(f"Stage 2: Creating {OUT_DIR} directory")
    print("----------------------------------")
    print("The following folder structure will be created:\n")
    print(f"{OUT_DIR}")
    print(f"└── {subdir}")
    print("    ├── images")
    print("    └── annotations\n")

    subdirectory = os.path.join(OUT_DIR, subdir)
    images_dir = os.path.join(subdirectory, "images")
    annotations_dir = os.path.join(subdirectory, "annotations")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(subdirectory, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)


def stage3(subdir):
    print(f"Stage 3: Copying files to {OUT_DIR} directory")
    print("------------------------------------------")

    filepath = os.path.join("temp", "stage1.txt")
    with open(filepath, "r") as f:
        filenames = [line.strip() for line in f.readlines()]

    for filename in tqdm(filenames):
        image_src = os.path.join(SOURCE_DIR, subdir, "images", filename + ".jpg")
        annotation_src = os.path.join(SOURCE_DIR, subdir, "annotations", filename + ".json")

        image_dst = os.path.join(OUT_DIR, subdir, "images", filename + ".jpg")
        annotation_dst = os.path.join(OUT_DIR, subdir, "annotations", filename + ".json")

        shutil.copyfile(image_src, image_dst)
        shutil.copyfile(annotation_src, annotation_dst)

    print()


def stage4(subdir):
    print("Stage 4: Cropping images to bounding box")
    print("----------------------------------------")

    annotations_dir = os.path.join(OUT_DIR, subdir, "annotations")
    images_dir = os.path.join(OUT_DIR, subdir, "images")
    cropped_images_dir = os.path.join(OUT_DIR, subdir, "cropped")
    os.makedirs(cropped_images_dir, exist_ok=True)

    for filename in tqdm(os.listdir(annotations_dir)):
        if filename.endswith(".json"):
            annotation_file = os.path.join(annotations_dir, filename)
            with open(annotation_file, "r") as f:
                data = json.load(f)

            bbox = extract_fields(data)["bbox"]
            image_index = os.path.splitext(filename)[0]
            image_path = os.path.join(images_dir, image_index + ".jpg")
            output_path = os.path.join(cropped_images_dir, image_index + ".jpg")

            crop_image(image_path, bbox, output_path)

    shutil.rmtree(images_dir)
    os.rename(cropped_images_dir, images_dir)

    print()


def stage5(subdir):
    print("Stage 5: Modifying annotation files")
    print("-----------------------------------")

    annotations_dir = os.path.join(OUT_DIR, subdir, "annotations")
    modified_annotations_dir = os.path.join(OUT_DIR, subdir, "modified")
    os.makedirs(modified_annotations_dir, exist_ok=True)

    for filename in tqdm(os.listdir(annotations_dir)):
        if filename.endswith(".json"):
            annotation_file = os.path.join(annotations_dir, filename)
            with open(annotation_file, "r") as f:
                data = json.load(f)

            fields = extract_fields(data)

            bbox = fields["bbox"]
            landmarks = fields["landmarks"]

            modified_landmarks = []
            for i in range(0, len(landmarks), 3):
                x = landmarks[i] - bbox[0] if landmarks[i + 2] != 0 else 0
                y = landmarks[i + 1] - bbox[1] if landmarks[i + 2] != 0 else 0
                v = landmarks[i + 2]

                if -5 < x < 0:
                    x = max(0, x)

                if -5 < y < 0:
                    y = max(0, y)

                modified_landmarks.append([x, y, v])

            modified_data = {
                "landmarks": modified_landmarks,
                "width": bbox[2] - bbox[0],
                "height": bbox[3] - bbox[1],
                "scale": fields["scale"],
                "occlusion": fields["occlusion"],
                "zoom": fields["zoom"],
                "viewpoint": fields["viewpoint"],
            }

            filepath = os.path.join(modified_annotations_dir, filename)

            with open(filepath, "w") as f:
                json.dump(modified_data, f, indent=4)

    shutil.rmtree(annotations_dir)
    os.rename(modified_annotations_dir, annotations_dir)

    print()


def stage6(subdir):
    print("Stage 6: Removing invalid files:")
    print(f"      1. Some landmarks are invisible")
    print(f"      2. Some landmarks are outside the image")
    print(f"      3. Images are smaller than {MIN_SIZE} px")
    print(f"      4. Images are larger than {MAX_SIZE} px")
    print(f"      5. Image width is larger than {MAX_WIDTH} px")
    print("--------------------------------------")

    annotations_dir = os.path.join(OUT_DIR, subdir, "annotations")

    for filename in tqdm(os.listdir(annotations_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(annotations_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            width = data["width"]
            height = data["height"]

            invalid_landmarks = any([point == [0, 0, 0] for point in data["landmarks"]]) or any(
                [point[0] < -5 or point[1] < -5 for point in data["landmarks"]]
            )
            invalid_size = width < MIN_SIZE or width > MAX_WIDTH or height < MIN_SIZE or height > MAX_SIZE

            if invalid_landmarks or invalid_size:
                os.remove(filepath)
                image_filename = os.path.splitext(filename)[0]
                image_filepath = os.path.join(OUT_DIR, subdir, "images", image_filename + ".jpg")
                os.remove(image_filepath)

    print()


def stage7(subdir):
    print("Stage 7: Renaming files")
    print("-----------------------")

    images_dir = os.path.join(OUT_DIR, subdir, "images")
    annotations_dir = os.path.join(OUT_DIR, subdir, "annotations")
    image_filenames = sorted(os.listdir(images_dir))
    annotation_filenames = sorted(os.listdir(annotations_dir))

    if len(image_filenames) != len(annotation_filenames):
        print("Error: Number of image files and annotation files do not match!")
        return

    for i, (old_image_filename, old_annotation_filename) in tqdm(
        enumerate(zip(image_filenames, annotation_filenames), start=1)
    ):
        new_image_filename = f"{i:06d}.jpg"
        new_annotation_filename = f"{i:06d}.json"

        old_image_filepath = os.path.join(images_dir, old_image_filename)
        new_image_filepath = os.path.join(images_dir, new_image_filename)
        old_annotation_filepath = os.path.join(annotations_dir, old_annotation_filename)
        new_annotation_filepath = os.path.join(annotations_dir, new_annotation_filename)

        os.rename(old_image_filepath, new_image_filepath)
        os.rename(old_annotation_filepath, new_annotation_filepath)

    shutil.rmtree("temp")

    print("----------------------")
    print("Processing successful!")
    print("----------------------")


def main():
    subdir = "train"
    stage1(subdir)
    stage2(subdir)
    stage3(subdir)
    stage4(subdir)
    stage5(subdir)
    stage6(subdir)
    stage7(subdir)

    subdir = "test"
    stage1(subdir)
    stage2(subdir)
    stage3(subdir)
    stage4(subdir)
    stage5(subdir)
    stage6(subdir)
    stage7(subdir)


if __name__ == "__main__":
    main()
