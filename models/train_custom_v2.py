import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


# ============================================================
# PROJECT PATHS
# ============================================================

VYRON_DIR = Path(
    r"C:\Users\Admin\Desktop\try\Vyron"
)

TRAINING_PROJECT_DIR = Path(
    r"C:\Users\Admin\Desktop\cusotm\TestBatCustom"
)

sys.path.insert(0, str(VYRON_DIR))


from models.custom_detector_v2 import (
    CustomCricketDetectorV2,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = (
    TRAINING_PROJECT_DIR
    / "dataset"
    / "final_dataset"
)

TRAIN_IMAGES = (
    DATASET_DIR
    / "images"
    / "train"
)

TRAIN_LABELS = (
    DATASET_DIR
    / "labels"
    / "train"
)

VAL_IMAGES = (
    DATASET_DIR
    / "images"
    / "val"
)

VAL_LABELS = (
    DATASET_DIR
    / "labels"
    / "val"
)

# IMPORTANT:
# New folder so the old V2 checkpoints are NOT reused.
RUNS_DIR = (
    TRAINING_PROJECT_DIR
    / "runs"
    / "custom_v2_fixed"
)

IMAGE_SIZE = 640

BATCH_SIZE = 2

EPOCHS = 30

LEARNING_RATE = 0.0005

NUM_CLASSES = 2

NUM_SLOTS = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class CricketDataset(Dataset):

    def __init__(
        self,
        image_dir,
        label_dir,
        image_size=640,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size

        self.image_paths = sorted(
            [
                p
                for p in self.image_dir.iterdir()
                if p.suffix.lower()
                in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                ]
            ]
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        image_size,
                        image_size,
                    )
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        label_path = (
            self.label_dir
            / f"{image_path.stem}.txt"
        )

        boxes = []
        classes = []

        if label_path.exists():

            with open(
                label_path,
                "r",
                encoding="utf-8",
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    if len(parts) != 5:
                        continue

                    try:
                        class_id = int(parts[0])

                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                    except ValueError:
                        continue

                    # Basic safety validation
                    if not (
                        0.0 <= x_center <= 1.0
                        and
                        0.0 <= y_center <= 1.0
                        and
                        0.0 <= width <= 1.0
                        and
                        0.0 <= height <= 1.0
                    ):
                        continue

                    if class_id not in [0, 1]:
                        continue

                    boxes.append(
                        [
                            x_center,
                            y_center,
                            width,
                            height,
                        ]
                    )

                    classes.append(
                        class_id
                    )

        if boxes:

            boxes = torch.tensor(
                boxes,
                dtype=torch.float32,
            )

            classes = torch.tensor(
                classes,
                dtype=torch.long,
            )

        else:

            boxes = torch.empty(
                (0, 4),
                dtype=torch.float32,
            )

            classes = torch.empty(
                (0,),
                dtype=torch.long,
            )

        target = {
            "boxes": boxes,
            "labels": classes,
        }

        return image, target


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    images = []
    targets = []

    for image, target in batch:

        images.append(image)
        targets.append(target)

    images = torch.stack(images)

    return images, targets


# ============================================================
# HELPER
# ============================================================

def clamp_probability(value, eps=1e-5):

    return torch.clamp(
        value,
        eps,
        1.0 - eps,
    )


def inverse_sigmoid(value):

    value = clamp_probability(value)

    return torch.log(
        value / (1.0 - value)
    )


# ============================================================
# FOCAL OBJECTNESS LOSS
# ============================================================

class FocalObjectnessLoss(nn.Module):

    def __init__(
        self,
        alpha=0.75,
        gamma=2.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits,
        targets,
    ):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = clamp_probability(
            probabilities
        )

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        positive_weight = (
            self.alpha
            * targets
            * torch.pow(
                1.0 - probabilities,
                self.gamma,
            )
        )

        negative_weight = (
            (1.0 - self.alpha)
            * (1.0 - targets)
            * torch.pow(
                probabilities,
                self.gamma,
            )
        )

        focal_weight = (
            positive_weight
            + negative_weight
        )

        loss = (
            focal_weight * bce
        )

        return loss.mean()


# ============================================================
# V2 LOSS
# ============================================================

class DetectionLossV2(nn.Module):

    def __init__(
        self,
        num_classes=2,
        num_slots=2,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_slots = num_slots

        self.class_loss = nn.CrossEntropyLoss()

        self.box_loss = nn.SmoothL1Loss(
            reduction="sum"
        )

        self.objectness = FocalObjectnessLoss(
            alpha=0.75,
            gamma=2.0,
        )

    def forward(
        self,
        predictions,
        targets,
    ):

        batch_size, channels, grid_h, grid_w = (
            predictions.shape
        )

        values_per_slot = (
            5 + self.num_classes
        )

        expected_channels = (
            self.num_slots
            * values_per_slot
        )

        if channels != expected_channels:

            raise RuntimeError(
                "Unexpected model output shape: "
                f"{tuple(predictions.shape)}. "
                f"Expected channels={expected_channels}."
            )

        predictions = predictions.view(
            batch_size,
            self.num_slots,
            values_per_slot,
            grid_h,
            grid_w,
        )

        device = predictions.device

        # ----------------------------------------------------
        # Objectness targets
        # ----------------------------------------------------

        object_target = torch.zeros(
            (
                batch_size,
                self.num_slots,
                grid_h,
                grid_w,
            ),
            dtype=torch.float32,
            device=device,
        )

        # ----------------------------------------------------
        # Loss accumulators
        # ----------------------------------------------------

        total_box_loss = torch.tensor(
            0.0,
            device=device,
        )

        total_class_loss = torch.tensor(
            0.0,
            device=device,
        )

        object_count = 0

        # ----------------------------------------------------
        # Assign objects
        # ----------------------------------------------------

        for batch_index in range(
            batch_size
        ):

            boxes = targets[
                batch_index
            ]["boxes"].to(device)

            labels = targets[
                batch_index
            ]["labels"].to(device)

            cell_slots = {}

            for object_index in range(
                boxes.shape[0]
            ):

                x_center = float(
                    boxes[
                        object_index,
                        0
                    ].item()
                )

                y_center = float(
                    boxes[
                        object_index,
                        1
                    ].item()
                )

                width = float(
                    boxes[
                        object_index,
                        2
                    ].item()
                )

                height = float(
                    boxes[
                        object_index,
                        3
                    ].item()
                )

                class_id = int(
                    labels[
                        object_index
                    ].item()
                )

                # ------------------------------------------------
                # Determine grid cell
                # ------------------------------------------------

                grid_x = min(
                    max(
                        int(
                            x_center
                            * grid_w
                        ),
                        0,
                    ),
                    grid_w - 1,
                )

                grid_y = min(
                    max(
                        int(
                            y_center
                            * grid_h
                        ),
                        0,
                    ),
                    grid_h - 1,
                )

                cell = (
                    grid_y,
                    grid_x,
                )

                used_slots = cell_slots.get(
                    cell,
                    [],
                )

                # Maximum two objects per cell
                if (
                    len(used_slots)
                    >= self.num_slots
                ):
                    continue

                slot = len(
                    used_slots
                )

                used_slots.append(
                    slot
                )

                cell_slots[cell] = (
                    used_slots
                )

                # ------------------------------------------------
                # Objectness target
                # ------------------------------------------------

                object_target[
                    batch_index,
                    slot,
                    grid_y,
                    grid_x,
                ] = 1.0

                prediction = predictions[
                    batch_index,
                    slot,
                    :,
                    grid_y,
                    grid_x,
                ]

                # ------------------------------------------------
                # LOCAL X / Y
                #
                # Decoder does:
                # sigmoid(tx), sigmoid(ty)
                #
                # Therefore training compares:
                # sigmoid(prediction) vs local target
                # ------------------------------------------------

                local_x = (
                    x_center
                    * grid_w
                    - grid_x
                )

                local_y = (
                    y_center
                    * grid_h
                    - grid_y
                )

                target_local_xy = torch.tensor(
                    [
                        local_x,
                        local_y,
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                predicted_xy = torch.sigmoid(
                    prediction[0:2]
                )

                total_box_loss += (
                    nn.functional.smooth_l1_loss(
                        predicted_xy,
                        target_local_xy,
                        reduction="sum",
                    )
                )

                # ------------------------------------------------
                # WIDTH / HEIGHT
                #
                # Decoder:
                # exp(tw) / grid_w
                #
                # Therefore target is:
                # log(width * grid_w)
                # ------------------------------------------------

                target_tw = torch.log(
                    torch.tensor(
                        max(
                            width * grid_w,
                            1e-4,
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                )

                target_th = torch.log(
                    torch.tensor(
                        max(
                            height * grid_h,
                            1e-4,
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                )

                target_wh = torch.stack(
                    [
                        target_tw,
                        target_th,
                    ]
                )

                predicted_wh = prediction[
                    2:4
                ]

                total_box_loss += (
                    self.box_loss(
                        predicted_wh,
                        target_wh,
                    )
                )

                # ------------------------------------------------
                # CLASSIFICATION
                # ------------------------------------------------

                class_logits = prediction[
                    5:
                ]

                target_class = torch.tensor(
                    [class_id],
                    dtype=torch.long,
                    device=device,
                )

                total_class_loss += (
                    self.class_loss(
                        class_logits.unsqueeze(0),
                        target_class,
                    )
                )

                object_count += 1

        # --------------------------------------------------------
        # Objectness loss
        # --------------------------------------------------------

        predicted_objectness = predictions[
            :,
            :,
            4,
            :,
            :,
        ]

        total_objectness_loss = (
            self.objectness(
                predicted_objectness,
                object_target,
            )
        )

        # --------------------------------------------------------
        # Normalize positive losses
        # --------------------------------------------------------

        if object_count > 0:

            total_box_loss /= object_count

            total_class_loss /= object_count

        # --------------------------------------------------------
        # Final loss
        # --------------------------------------------------------

        total_loss = (
            total_objectness_loss
            + 5.0 * total_box_loss
            + total_class_loss
        )

        return (
            total_loss,
            total_objectness_loss.detach(),
            total_box_loss.detach(),
            total_class_loss.detach(),
        )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    dataloader,
    criterion,
):

    model.eval()

    total_loss = 0.0

    batch_count = 0

    with torch.no_grad():

        for images, targets in dataloader:

            images = images.to(
                DEVICE
            )

            predictions = model(
                images
            )

            (
                loss,
                _,
                _,
                _,
            ) = criterion(
                predictions,
                targets,
            )

            total_loss += (
                loss.item()
            )

            batch_count += 1

    if batch_count == 0:

        return 0.0

    return (
        total_loss
        / batch_count
    )


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 70)
    print(
        "CUSTOM CRICKET DETECTOR V2 - FIXED TRAINING"
    )
    print("=" * 70)

    print(
        f"Training project : "
        f"{TRAINING_PROJECT_DIR}"
    )

    print(
        f"Dataset          : "
        f"{DATASET_DIR}"
    )

    print(
        f"Output directory : "
        f"{RUNS_DIR}"
    )

    print(
        f"Device            : "
        f"{DEVICE}"
    )

    print(
        f"Image size        : "
        f"{IMAGE_SIZE}"
    )

    print(
        f"Batch size        : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs            : "
        f"{EPOCHS}"
    )

    print(
        f"Learning rate     : "
        f"{LEARNING_RATE}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    required_paths = [
        TRAIN_IMAGES,
        TRAIN_LABELS,
        VAL_IMAGES,
        VAL_LABELS,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = CricketDataset(
        TRAIN_IMAGES,
        TRAIN_LABELS,
        IMAGE_SIZE,
    )

    val_dataset = CricketDataset(
        VAL_IMAGES,
        VAL_LABELS,
        IMAGE_SIZE,
    )

    print(
        f"Training images   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images : "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CustomCricketDetectorV2(
        num_classes=NUM_CLASSES,
        num_slots=NUM_SLOTS,
    ).to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters  : "
        f"{total_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = DetectionLossV2(
        num_classes=NUM_CLASSES,
        num_slots=NUM_SLOTS,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Fresh training.
    # We DO NOT load old custom_v2 checkpoints.
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    best_path = (
        RUNS_DIR
        / "custom_detector_v2_fixed_best.pt"
    )

    latest_path = (
        RUNS_DIR
        / "custom_detector_v2_fixed_latest.pt"
    )

    print()
    print(
        "Starting NEW V2 FIXED training."
    )

    print(
        "Old V2 checkpoints will NOT be loaded."
    )

    print()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        running_objectness = 0.0

        running_box = 0.0

        running_class = 0.0

        batch_count = 0

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for images, targets in train_loader:

            images = images.to(
                DEVICE
            )

            optimizer.zero_grad()

            predictions = model(
                images
            )

            (
                loss,
                objectness_loss,
                box_loss,
                class_loss,
            ) = criterion(
                predictions,
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
            )

            running_objectness += (
                objectness_loss.item()
            )

            running_box += (
                box_loss.item()
            )

            running_class += (
                class_loss.item()
            )

            batch_count += 1

        # ----------------------------------------------------
        # Average training losses
        # ----------------------------------------------------

        train_loss = (
            running_loss
            / max(
                batch_count,
                1,
            )
        )

        train_objectness = (
            running_objectness
            / max(
                batch_count,
                1,
            )
        )

        train_box = (
            running_box
            / max(
                batch_count,
                1,
            )
        )

        train_class = (
            running_class
            / max(
                batch_count,
                1,
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = validate(
            model,
            val_loader,
            criterion,
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print()
        print(
            "-" * 70
        )

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss      : "
            f"{train_loss:.6f}"
        )

        print(
            f"Objectness Loss : "
            f"{train_objectness:.6f}"
        )

        print(
            f"Box Loss        : "
            f"{train_box:.6f}"
        )

        print(
            f"Class Loss      : "
            f"{train_class:.6f}"
        )

        print(
            f"Val Loss        : "
            f"{val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Save latest
        # ----------------------------------------------------

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "train_loss":
                    train_loss,
                "val_loss":
                    val_loss,
            },
            latest_path,
        )

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "train_loss":
                        train_loss,
                    "val_loss":
                        val_loss,
                },
                best_path,
            )

            print()
            print(
                "BEST MODEL SAVED:"
            )

            print(
                best_path
            )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "V2 FIXED TRAINING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Best validation loss : "
        f"{best_val_loss:.6f}"
    )

    print()
    print(
        "Best model:"
    )

    print(
        best_path
    )

    print()
    print(
        "Latest model:"
    )

    print(
        latest_path
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()