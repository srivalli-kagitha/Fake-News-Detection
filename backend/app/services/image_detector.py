"""Real vs AI-generated / manipulated image detection + Grad-CAM explanation.

Model: frozen CLIP ViT-L/14 image encoder + a small trained linear head
(training/train_image_detector.py -> backend/models/image_detector/{config.json, head.pt}).
"""
import base64
import io
import json

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoConfig, CLIPImageProcessor, CLIPVisionModelWithProjection

from ..config import get_settings, resolve_local
from .common import Lazy, ModelNotReadyError, get_device

FAKE_NAMES = {"fake", "ai", "ai-generated", "ai_generated", "synthetic", "manipulated", "generated"}


class ClipDetector(nn.Module):
    """Keep this class identical to the one in training/train_image_detector.py."""

    def __init__(self, clip_name: str, num_labels: int):
        super().__init__()
        # Build the vision config ourselves: some transformers installs pass the full CLIPConfig
        # to the vision model, which then fails with "'CLIPConfig' object has no attribute 'hidden_size'".
        full = AutoConfig.from_pretrained(clip_name)
        vision = getattr(full, "vision_config", full)
        if hasattr(full, "projection_dim"):
            vision.projection_dim = full.projection_dim
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_name, config=vision)
        self.head = nn.Linear(self.clip.config.projection_dim, num_labels)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        emb = self.clip(pixel_values=pixel_values).image_embeds
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return self.head(emb)


class ImageDetector:
    def __init__(self, model_dir: str):
        path = resolve_local(model_dir)
        config_file, head_file = path / "config.json", path / "head.pt"
        if not (config_file.exists() and head_file.exists()):
            raise ModelNotReadyError(
                "The image detector is not trained yet. Train it with training/train_image_detector.py "
                f"and copy the output folder to backend/{model_dir} (see training/README.md)."
            )
        config = json.loads(config_file.read_text(encoding="utf-8"))
        self.labels: list[str] = config["labels"]
        self.clip_name: str = config["clip_model"]
        self.device = get_device()

        self.model = ClipDetector(self.clip_name, len(self.labels))
        self.model.head.load_state_dict(torch.load(head_file, map_location="cpu"))
        self.model.to(self.device).eval()

        processor = CLIPImageProcessor.from_pretrained(self.clip_name)
        self.size = int(processor.crop_size["height"])
        self.mean = np.array(processor.image_mean, dtype=np.float32)
        self.std = np.array(processor.image_std, dtype=np.float32)
        vcfg = self.model.clip.config
        self.grid = vcfg.image_size // vcfg.patch_size
        self.fake_index = next(
            (i for i, n in enumerate(self.labels) if n.lower() in FAKE_NAMES), len(self.labels) - 1
        )

    # ---------- preprocessing ----------
    def _prepare(self, image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
        image = image.convert("RGB")
        w, h = image.size
        scale = self.size / min(w, h)
        image = image.resize((max(self.size, round(w * scale)), max(self.size, round(h * scale))), Image.BICUBIC)
        w, h = image.size
        left, top = (w - self.size) // 2, (h - self.size) // 2
        image = image.crop((left, top, left + self.size, top + self.size))
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device), rgb

    def _display_label(self, idx: int) -> str:
        return "AI-Generated / Manipulated" if idx == self.fake_index else "Real"

    def _result(self, probs: np.ndarray) -> dict:
        idx = int(probs.argmax())
        return {
            "label": self._display_label(idx),
            "confidence": float(probs[idx]),
            "probabilities": {self._display_label(i): float(p) for i, p in enumerate(probs)},
            "fake_probability": float(probs[self.fake_index]),
        }

    # ---------- prediction ----------
    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        tensor, _ = self._prepare(image)
        probs = torch.softmax(self.model(tensor).float(), dim=-1)[0].cpu().numpy()
        return self._result(probs)

    def predict_with_gradcam(self, image: Image.Image) -> dict:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        tensor, rgb = self._prepare(image)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor).float(), dim=-1)[0].cpu().numpy()
        result = self._result(probs)
        target_idx = int(probs.argmax())

        grid = self.grid

        def reshape_transform(t: torch.Tensor) -> torch.Tensor:
            t = t[:, 1:, :]  # drop CLS token
            return t.reshape(t.size(0), grid, grid, t.size(2)).permute(0, 3, 1, 2)

        target_layer = self.model.clip.vision_model.encoder.layers[-1].layer_norm1
        with GradCAM(model=self.model, target_layers=[target_layer], reshape_transform=reshape_transform) as cam:
            heatmap = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_idx)])[0]
        overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)

        result["explanation"] = {
            "method": "Grad-CAM",
            "explained_label": self._display_label(target_idx),
            "overlay": _to_data_url(overlay),
            "original": _to_data_url((rgb * 255).astype(np.uint8)),
        }
        return result


def _to_data_url(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


_settings = get_settings()
_detector = Lazy("Image detector", lambda: ImageDetector(_settings.image_model))


def get_image_detector() -> ImageDetector:
    return _detector.get()


def image_model_status() -> str:
    if _detector.loaded:
        return "loaded"
    path = resolve_local(_settings.image_model)
    return "ready" if (path / "config.json").exists() and (path / "head.pt").exists() else "not_trained"
