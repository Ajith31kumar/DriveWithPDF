import torch
import torch.nn as nn


# ============================================================
# CUSTOM CRICKET DETECTOR V2
# ============================================================
#
# Classes:
#   0 = Bat
#   1 = Ball
#
# Design:
#   - 80x80 detection grid for 640x640 input
#   - 2 prediction slots per grid cell
#   - Correct local x/y coordinate encoding
#   - Log-space width/height encoding
#
# Output per slot:
#   tx, ty, tw, th, objectness, class_0, class_1
#
# Total output channels:
#   2 slots * (5 + 2 classes) = 14
# ============================================================


NUM_CLASSES = 2
NUM_SLOTS = 2
IMAGE_SIZE = 640


# ============================================================
# CONV BLOCK
# ============================================================

class ConvBlock(nn.Module):
    """
    Conv2d -> BatchNorm -> SiLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# BACKBONE
# ============================================================

class BackboneV2(nn.Module):
    """
    Lightweight backbone.

    640x640 input
        -> 320x320
        -> 160x160
        -> 80x80

    We intentionally stop at 80x80 so the cricket ball
    retains more spatial information.
    """

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            ConvBlock(3, 32, stride=2),
            ConvBlock(32, 32),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 64),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


# ============================================================
# DETECTION HEAD
# ============================================================

class DetectionHeadV2(nn.Module):
    """
    Two-slot anchor-free style detection head.

    Each grid cell has two independent prediction slots.

    This allows:
        slot 0 -> Bat
        slot 1 -> Ball

    when both objects are close to the same grid cell.
    """

    def __init__(
        self,
        in_channels,
        num_classes=NUM_CLASSES,
        num_slots=NUM_SLOTS,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_slots = num_slots

        outputs_per_slot = 5 + num_classes

        self.prediction = nn.Conv2d(
            in_channels,
            num_slots * outputs_per_slot,
            kernel_size=1,
        )

    def forward(self, x):
        return self.prediction(x)


# ============================================================
# MODEL
# ============================================================

class CustomCricketDetectorV2(nn.Module):
    """
    Custom Bat + Ball Detector V2.

    Classes:
        0 = Bat
        1 = Ball

    Input:
        [B, 3, 640, 640]

    Output:
        [B, 14, 80, 80]
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        num_slots=NUM_SLOTS,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_slots = num_slots

        self.backbone = BackboneV2()

        self.head = DetectionHeadV2(
            in_channels=128,
            num_classes=num_classes,
            num_slots=num_slots,
        )

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)

        return predictions


# ============================================================
# BOX DECODER
# ============================================================

def decode_predictions_v2(
    predictions,
    image_size=IMAGE_SIZE,
):
    """
    Decode V2 predictions.

    Input:
        predictions:
            [B, 14, H, W]

    Returns:
        boxes:
            [B, H, W, S, 4]
            normalized xyxy

        objectness:
            [B, H, W, S]

        class_probs:
            [B, H, W, S, C]
    """

    batch_size, channels, grid_h, grid_w = predictions.shape

    expected_channels = NUM_SLOTS * (
        5 + NUM_CLASSES
    )

    if channels != expected_channels:
        raise ValueError(
            "Unexpected prediction channel count. "
            f"Expected {expected_channels}, got {channels}."
        )

    outputs_per_slot = 5 + NUM_CLASSES

    predictions = predictions.permute(
        0,
        2,
        3,
        1,
    ).contiguous()

    predictions = predictions.view(
        batch_size,
        grid_h,
        grid_w,
        NUM_SLOTS,
        outputs_per_slot,
    )

    # --------------------------------------------------------
    # Raw values
    # --------------------------------------------------------

    raw_xy = predictions[..., 0:2]
    raw_wh = predictions[..., 2:4]

    raw_objectness = predictions[..., 4]

    class_logits = predictions[..., 5:]

    # --------------------------------------------------------
    # Objectness
    # --------------------------------------------------------

    objectness = torch.sigmoid(
        raw_objectness
    )

    # --------------------------------------------------------
    # Class probabilities
    # --------------------------------------------------------

    class_probs = torch.softmax(
        class_logits,
        dim=-1,
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    device = predictions.device

    y_grid, x_grid = torch.meshgrid(
        torch.arange(
            grid_h,
            device=device,
            dtype=torch.float32,
        ),
        torch.arange(
            grid_w,
            device=device,
            dtype=torch.float32,
        ),
        indexing="ij",
    )

    x_grid = x_grid.view(
        1,
        grid_h,
        grid_w,
        1,
    )

    y_grid = y_grid.view(
        1,
        grid_h,
        grid_w,
        1,
    )

    # --------------------------------------------------------
    # Local x/y -> global normalized x/y
    # --------------------------------------------------------

    center_x = (
        torch.sigmoid(raw_xy[..., 0])
        + x_grid
    ) / grid_w

    center_y = (
        torch.sigmoid(raw_xy[..., 1])
        + y_grid
    ) / grid_h

    # --------------------------------------------------------
    # Width / Height
    #
    # Training will use:
    #
    # tw = log(width * grid_w)
    # th = log(height * grid_h)
    #
    # Decode:
    #
    # width = exp(tw) / grid_w
    # --------------------------------------------------------

    width = (
        torch.exp(
            torch.clamp(
                raw_wh[..., 0],
                min=-6.0,
                max=6.0,
            )
        )
        / grid_w
    )

    height = (
        torch.exp(
            torch.clamp(
                raw_wh[..., 1],
                min=-6.0,
                max=6.0,
            )
        )
        / grid_h
    )

    # --------------------------------------------------------
    # xywh -> xyxy
    # --------------------------------------------------------

    x1 = center_x - width / 2.0
    y1 = center_y - height / 2.0

    x2 = center_x + width / 2.0
    y2 = center_y + height / 2.0

    boxes = torch.stack(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        dim=-1,
    )

    boxes = boxes.clamp(
        0.0,
        1.0,
    )

    return (
        boxes,
        objectness,
        class_probs,
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model_v2():
    return CustomCricketDetectorV2(
        num_classes=NUM_CLASSES,
        num_slots=NUM_SLOTS,
    )


# ============================================================
# MODEL CHECK
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CUSTOM CRICKET DETECTOR V2")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device : {device}")

    model = build_model_v2().to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_params:,}"
    )

    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    with torch.no_grad():
        output = model(
            dummy_input
        )

    print(
        f"Input shape  : "
        f"{tuple(dummy_input.shape)}"
    )

    print(
        f"Output shape : "
        f"{tuple(output.shape)}"
    )

    boxes, objectness, class_probs = (
        decode_predictions_v2(
            output,
            image_size=IMAGE_SIZE,
        )
    )

    print(
        f"Boxes shape       : "
        f"{tuple(boxes.shape)}"
    )

    print(
        f"Objectness shape  : "
        f"{tuple(objectness.shape)}"
    )

    print(
        f"Class probs shape : "
        f"{tuple(class_probs.shape)}"
    )

    print("=" * 70)
    print("MODEL CHECK PASSED")
    print("=" * 70)