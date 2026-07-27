"""Arm inpainting, replicating HumanEgo (https://github.com/TX-Leo/HumanEgo).

Pipeline, per frame:

1. **Grounding DINO** — open-vocabulary detection with the text prompt
   ``"arm. hand."`` -> candidate boxes for the demonstrator's arm/hand.
2. **SAM2** — promptable segmentation from those boxes -> a binary mask; the
   per-box masks are unioned into one arm+hand mask.
3. **dilate** the mask to cover soft edges / shadows (HumanEgo's ``Lama.py`` does
   the same before inpainting).
4. **LaMa** — inpaint the masked region so the arm is painted out, leaving a
   clean, embodiment-agnostic frame.

Grounding DINO and SAM2 come from ``transformers``; LaMa is
``simple-lama-inpainting`` (Big-LaMa). Models load lazily on the first cache
miss, so a fully-cached training run never pays the load cost.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .base import VisualMethod, register_visual_method


@register_visual_method("arm_inpaint")
class ArmInpaintMethod(VisualMethod):
    """Detect + segment + inpaint the human arm/hand out of a frame.

    Args:
        prompt: Grounding DINO text query. Lowercase, phrases separated by ".".
        box_threshold: detection box confidence threshold. Kept low (0.2) on
            purpose: DINO scores the compact *hand* box high (~0.5-0.65) but the
            full *arm* box only ~0.22-0.30, so a higher threshold keeps just the
            hand and leaves the forearm un-inpainted.
        text_threshold: detection text-match threshold.
        dilation: mask dilation kernel size in px (0 disables). Widens the mask
            so LaMa also repaints the arm's soft boundary.
        max_boxes: keep at most this many highest-scoring boxes.
        min_box_area_frac / max_box_area_frac: drop boxes whose area (as a
            fraction of the frame) is outside this range — filters spurious
            specks and whole-frame false positives.
        dino_model_id / sam2_model_id: HF model ids.
        device: torch device (default: cuda if available else cpu).
    """

    def __init__(
        self,
        prompt: str = "arm. hand.",
        box_threshold: float = 0.2,
        text_threshold: float = 0.2,
        dilation: int = 15,
        max_boxes: int = 8,
        min_box_area_frac: float = 0.0008,
        max_box_area_frac: float = 0.6,
        dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam2_model_id: str = "facebook/sam2-hiera-small",
        device: str | None = None,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.dilation = dilation
        self.max_boxes = max_boxes
        self.min_box_area_frac = min_box_area_frac
        self.max_box_area_frac = max_box_area_frac
        self.dino_model_id = dino_model_id
        self.sam2_model_id = sam2_model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def config_signature(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "dilation": self.dilation,
            "max_boxes": self.max_boxes,
            "min_box_area_frac": self.min_box_area_frac,
            "max_box_area_frac": self.max_box_area_frac,
            "dino_model_id": self.dino_model_id,
            "sam2_model_id": self.sam2_model_id,
        }

    # --- model loading -----------------------------------------------------

    def _ensure_models(self) -> None:
        if self._loaded:
            return
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
        )
        from simple_lama_inpainting import SimpleLama

        self._dino_processor = AutoProcessor.from_pretrained(self.dino_model_id)
        self._dino = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_model_id)
            .eval()
            .to(self.device)
        )
        self._sam2_processor = Sam2Processor.from_pretrained(self.sam2_model_id)
        self._sam2 = Sam2Model.from_pretrained(self.sam2_model_id).eval().to(self.device)
        self._lama = SimpleLama(device=torch.device(self.device))
        self._loaded = True

    # --- detection + segmentation -----------------------------------------

    def detect_boxes(self, image: np.ndarray) -> np.ndarray:
        """Grounding DINO -> ``(N, 4)`` xyxy pixel boxes, score-sorted & filtered."""
        self._ensure_models()
        pil = Image.fromarray(image)
        h, w = image.shape[:2]
        inputs = self._dino_processor(images=pil, text=self.prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._dino(**inputs)
        results = self._dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]
        boxes = results["boxes"].detach().cpu().numpy().reshape(-1, 4)
        scores = results["scores"].detach().cpu().numpy().reshape(-1)

        frame_area = float(h * w)
        keep = []
        for box, score in zip(boxes, scores):
            x0, y0, x1, y1 = box
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            frac = area / frame_area
            if self.min_box_area_frac <= frac <= self.max_box_area_frac:
                keep.append((score, box))
        keep.sort(key=lambda t: t[0], reverse=True)
        keep = keep[: self.max_boxes]
        if not keep:
            return np.zeros((0, 4), dtype=np.float32)
        return np.stack([b for _, b in keep]).astype(np.float32)

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Return a ``HxW`` uint8 mask (255 = arm/hand) for ``image`` (RGB uint8)."""
        h, w = image.shape[:2]
        boxes = self.detect_boxes(image)
        if len(boxes) == 0:
            return np.zeros((h, w), dtype=np.uint8)

        pil = Image.fromarray(image)
        inputs = self._sam2_processor(
            images=pil,
            input_boxes=[boxes.tolist()],
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self._sam2(**inputs, multimask_output=False)
        masks = self._sam2_processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"]
        )[0]  # (num_boxes, 1, H, W)
        masks = masks.detach().cpu().numpy()
        union = np.any(masks > 0.5, axis=(0, 1)).astype(np.uint8) * 255

        if self.dilation and self.dilation > 0:
            k = int(self.dilation)
            kernel = np.ones((k, k), np.uint8)
            union = cv2.dilate(union, kernel, iterations=2)
        return union

    # --- inpainting --------------------------------------------------------

    def apply(self, image: np.ndarray) -> np.ndarray:
        mask = self.segment(image)
        if mask.sum() == 0:  # nothing detected -> frame unchanged
            return image
        self._ensure_models()
        result = self._lama(Image.fromarray(image), Image.fromarray(mask))
        result = np.asarray(result.convert("RGB"))
        if result.shape[:2] != image.shape[:2]:
            result = cv2.resize(result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return result

    def visualize(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Original / detected-boxes / mask overlay / inpainted result."""
        mask = self.segment(image)
        boxes = self.detect_boxes(image)

        boxed = image.copy()
        for x0, y0, x1, y1 in boxes.astype(int):
            cv2.rectangle(boxed, (x0, y0), (x1, y1), (0, 255, 0), 2)

        overlay = image.copy()
        red = np.zeros_like(image)
        red[..., 0] = 255
        m = mask > 0
        overlay[m] = (0.5 * overlay[m] + 0.5 * red[m]).astype(np.uint8)

        if mask.sum() == 0:
            result = image
        else:
            self._ensure_models()
            result = np.asarray(self._lama(Image.fromarray(image), Image.fromarray(mask)).convert("RGB"))
            if result.shape[:2] != image.shape[:2]:
                result = cv2.resize(result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        return {"boxes": boxed, "mask_overlay": overlay, "result": result}
