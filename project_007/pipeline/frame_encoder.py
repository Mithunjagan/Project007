"""
PROJECT 007 — Frame Feature Encoder
Lightweight MobileNetV3-Small backbone for 576-dim frame embeddings.

Extracts appearance features from BGR crops for downstream classifiers
(LSTM violence detector, etc.). Runs on CUDA with FP16 when available.
"""

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Graceful import of torch / torchvision ───────────────
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning(
        "torch / torchvision not installed — FrameEncoder will be unavailable. "
        "Install with: pip install torch torchvision"
    )

# ── Device from config (with safe fallback) ──────────────
try:
    from config import DL_ENCODER_DEVICE
except ImportError:
    DL_ENCODER_DEVICE = None


# ImageNet normalization constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class FrameEncoder:
    """
    Extracts a 576-dim feature vector from a BGR frame using
    MobileNetV3-Small (pretrained on ImageNet) with classification
    head removed.

    Parameters
    ----------
    device : str, optional
        Force a specific device ('cuda' or 'cpu').
        Defaults to ``DL_ENCODER_DEVICE`` from config, falling back
        to 'cuda' if available, else 'cpu'.
    """

    def __init__(self, device: str = None):
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "FrameEncoder requires torch and torchvision. "
                "Install with: pip install torch torchvision"
            )

        # ── Resolve device ────────────────────────────────
        if device is not None:
            requested = device
        elif DL_ENCODER_DEVICE is not None:
            requested = DL_ENCODER_DEVICE
        else:
            requested = "cuda"

        if requested == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self._half = False
            logger.warning(
                "CUDA requested but not available. "
                "Falling back to CPU (FP16 disabled)."
            )
        else:
            self.device = torch.device(requested)
            self._half = self.device.type == "cuda"

        # ── Load model ────────────────────────────────────
        logger.info("Loading MobileNetV3-Small encoder…")
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier = nn.Identity()  # drop classification head → 576-dim
        model.eval()
        model.to(self.device)

        if self._half:
            model.half()
            logger.info("FP16 (half-precision) enabled on CUDA")

        self._model = model

        # ── Preprocessing pipeline ────────────────────────
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        logger.info(
            f"FrameEncoder ready — device={self.device}, "
            f"half={self._half}, output_dim=576"
        )

    # ──────────────────────────────────────────────────────
    def encode(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode a single BGR frame into a 576-dim feature vector.

        Parameters
        ----------
        frame : np.ndarray
            HxWx3 BGR image (OpenCV format).

        Returns
        -------
        np.ndarray
            1-D float32 array of shape ``(576,)``.
        """
        # BGR → RGB for torchvision transforms
        rgb = frame[:, :, ::-1].copy()

        tensor = self._transform(rgb).unsqueeze(0).to(self.device)
        if self._half:
            tensor = tensor.half()

        with torch.no_grad():
            features = self._model(tensor)

        return features.squeeze(0).cpu().float().numpy()
