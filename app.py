import argparse
from timeit import default_timer as timer
from ui.windows.window import Window, get_app, wait

TRUE_WIDTH = 60
TRUE_HEIGHT = 60
PIXELTOINCH = 10
MARGIN = 25
THRESHOLD = 25

CHECKPOINT = "lib/checkpoints/epoch_50.pth"
TRAIN_DATASET = "data/shirts/train"
TEST_DATASET = "data/shirts/test"


class Timer:
    def __init__(self, msg):
        print(msg)

    def __enter__(self):
        self.start = timer()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end = timer()
        runtime = self.end - self.start
        print(f"Took {runtime:.3f} sec\n")


class TeeSize(Window):
    def __init__(self, model, true_width, true_height, pixeltoinch, margin):
        super().__init__(true_width, true_height, pixeltoinch, margin)
        self.model = model

    @wait
    def start(self, signal):
        image_file = self.get_image_file()
        if image_file == "":
            return

        image_basename, _ = os.path.splitext(os.path.basename(image_file))
        image = cv2.imread(image_file)

        image = perspective_correction(image, self.true_width, self.true_height, self.pixeltoinch, self.margin)
        height, width = image.shape[:2]
        original_size = width if width > height else height
        scale_factor = original_size / 256

        os.makedirs("debug", exist_ok=True)

        savepath = os.path.join("debug", f"{image_basename}_corrected.png")
        print(f"[*] Perspective corrected image saved as: {savepath}")
        cv2.imwrite(savepath, image)

        image, landmarks, inference_time = predict(savepath, self.model)
        print(f"[*] Inference time: {inference_time * 1000:.3f} ms")

        savepath = os.path.join("debug", f"{image_basename}_landmarks.png")
        draw_landmarks(image, landmarks, savepath)
        print(f"[*] Predicted landmarks saved as: {savepath}")

        sizes, lines = measure(landmarks, scale_factor, self.pixeltoinch)

        savepath = os.path.join("debug", f"{image_basename}_measurement.png")
        draw_measurements(image, landmarks, lines, savepath)
        print(f"[*] Measurement lines saved as: {savepath}\n")

        self.update_sizes(sizes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TeeSize",
        description="%(prog)s - Automatically measure T-shirt sizes using deep learning",
        usage="python app.py [OPTIONS]",
        add_help=True,
        exit_on_error=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="gui",
        choices=["gui", "train", "test"],
        required=False,
        help="Select between launching application, training model, or testing model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT,
        required=False,
        help="Checkpoint file to use for inference, fine-tuning, or testing model",
    )
    parser.add_argument(
        "--dataset", type=str, required=False, help="Path to the dataset to use for training model or testing model"
    )
    parser.add_argument(
        "--truewidth",
        type=float,
        default=TRUE_WIDTH,
        required=False,
        help="Default value for the true width between chessboard patterns (in inches)",
    )
    parser.add_argument(
        "--trueheight",
        type=float,
        default=TRUE_HEIGHT,
        required=False,
        help="Default value for the true height between chessboard patterns (in inches)",
    )
    parser.add_argument(
        "--pixeltoinch",
        type=float,
        default=PIXELTOINCH,
        required=False,
        help="Number of pixels that should represent one inch in the perspective corrected image (does not affect measurements)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=MARGIN,
        required=False,
        help="Number of pixels by which to trim the perspective corrected image from each side",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        required=False,
        help="Threshold to use for calculating the percentage of correct keypoints (PCK)",
    )
    args = parser.parse_args()

    if args.mode == "gui":
        checkpoint_file = args.checkpoint
        true_width = args.truewidth
        true_height = args.trueheight
        pixeltoinch = args.pixeltoinch
        margin = args.margin

        with Timer(msg="Loading libraries ..."):
            import os
            import sys
            import cv2

            from lib.inference import predict, load_model
            from lib.image import perspective_correction, measure, draw_measurements
            from lib.utils import draw_landmarks

        with Timer(msg=f"Loading model: {checkpoint_file}"):
            model = load_model(checkpoint_file)

        with Timer(msg="Starting application ..."):
            app = get_app(sys.argv)
            teesize = TeeSize(model, true_width, true_height, pixeltoinch, margin)
            teesize.show()

        sys.exit(app.exec_())

    if args.mode == "train":
        with Timer(msg="Loading libraries ..."):
            from lib.train import train

        checkpoint_file = args.checkpoint
        dataset = args.dataset
        if dataset is None:
            dataset = TRAIN_DATASET

        train(dataset, checkpoint_file)

    if args.mode == "test":
        with Timer(msg="Loading libraries ..."):
            from lib.inference import load_model
            from lib.test import evaluate

        threshold = args.threshold
        checkpoint_file = args.checkpoint
        dataset = args.dataset
        if dataset is None:
            dataset = TEST_DATASET

        with Timer(msg=f"Loading model: {checkpoint_file}"):
            model = load_model(checkpoint_file)
            evaluate(dataset, model, threshold=threshold)
