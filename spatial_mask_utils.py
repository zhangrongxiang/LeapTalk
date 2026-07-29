"""Spatial mask extraction using MediaPipe for weighted ViBT loss."""
import cv2
import numpy as np
import torch
from typing import Iterable, Tuple

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARN] MediaPipe not available. Install with: pip install mediapipe")


def _connections_to_indices(connections: Iterable[tuple[int, int]]) -> list[int]:
    indices: set[int] = set()
    for src_idx, dst_idx in connections:
        indices.add(int(src_idx))
        indices.add(int(dst_idx))
    return sorted(indices)


class FaceLipMaskExtractor:
    """Extract face and lip masks from video frames using MediaPipe."""

    def __init__(self, device="cpu", reuse_previous_on_miss: bool = True):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required. Install with: pip install mediapipe")

        self.device = device
        self.reuse_previous_on_miss = reuse_previous_on_miss
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Use MediaPipe's official face contour definitions instead of broad manual ranges.
        self.LIPS_INDICES = _connections_to_indices(self.mp_face_mesh.FACEMESH_LIPS)
        self.FACE_OVAL_INDICES = _connections_to_indices(self.mp_face_mesh.FACEMESH_FACE_OVAL)

    @staticmethod
    def _kernel(size: int) -> np.ndarray | None:
        size = int(size)
        if size <= 0:
            return None
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    @staticmethod
    def _points_from_landmarks(landmarks, indices: list[int], width: int, height: int) -> np.ndarray:
        points: list[list[int]] = []
        for idx in indices:
            if idx >= len(landmarks.landmark):
                continue
            lm = landmarks.landmark[idx]
            x = int(np.clip(round(lm.x * (width - 1)), 0, width - 1))
            y = int(np.clip(round(lm.y * (height - 1)), 0, height - 1))
            points.append([x, y])
        if len(points) < 3:
            return np.empty((0, 2), dtype=np.int32)
        return np.asarray(points, dtype=np.int32)

    def _build_region_mask(
        self,
        points: np.ndarray,
        height: int,
        width: int,
        dilate_size: int,
        close_size: int,
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(points) < 3:
            return mask

        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 1)

        close_kernel = self._kernel(close_size)
        if close_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        dilate_kernel = self._kernel(dilate_size)
        if dilate_kernel is not None:
            mask = cv2.dilate(mask, dilate_kernel)
        return mask

    def extract_masks(
        self,
        frames: np.ndarray,  # ideally [T, H, W, 3] uint8 RGB (but this function is robust)
        dilate_lip: int = 5,
        dilate_face: int = 10,
        close_lip: int = 3,
        close_face: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract face and lip masks from video frames.

        Args:
            frames: numpy array [T, H, W, C] in RGB, uint8
            dilate_lip: dilation kernel size for lip mask
            dilate_face: dilation kernel size for face mask

        Returns:
            face_mask: [T, H, W] float tensor on device
            lip_mask: [T, H, W] float tensor on device
        """
        # Normalize input to numpy array
        if isinstance(frames, torch.Tensor):
            frames_np = frames.detach().cpu().numpy()
        else:
            frames_np = np.asarray(frames)

        # Normalize layout to [T, H, W, C]
        if frames_np.ndim == 3:
            # [T, H, W] -> add channel
            frames_np = frames_np[..., None]
        elif frames_np.ndim == 4:
            # Heuristic: if second dim looks like channels, convert [T, C, H, W] -> [T, H, W, C]
            if frames_np.shape[1] in (1, 3, 4) and frames_np.shape[-1] not in (1, 3, 4):
                frames_np = np.transpose(frames_np, (0, 2, 3, 1))
        else:
            raise ValueError(f"Unsupported frames ndim={frames_np.ndim}, shape={frames_np.shape}")

        T, H, W, C = frames_np.shape

        # Normalize channels to 3
        if C == 1:
            frames_np = np.repeat(frames_np, 3, axis=-1)
        elif C >= 4:
            frames_np = frames_np[..., :3]
        elif C != 3:
            raise ValueError(f"Input frames must have 1/3/4 channels, got C={C}")

        # Normalize dtype/range to uint8
        if frames_np.dtype != np.uint8:
            # Common training format: float in [-1, 1]
            if np.issubdtype(frames_np.dtype, np.floating):
                vmin = float(np.nanmin(frames_np))
                vmax = float(np.nanmax(frames_np))
                if vmin >= -1.1 and vmax <= 1.1:
                    frames_np = ((frames_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                else:
                    frames_np = frames_np.clip(0, 255).astype(np.uint8)
            else:
                frames_np = frames_np.astype(np.uint8)

        # Ensure contiguous memory for MediaPipe
        frames_np = np.ascontiguousarray(frames_np)
        face_masks = []
        lip_masks = []
        prev_face_mask = None
        prev_lip_mask = None

        for t in range(T):
            frame = frames_np[t]
            results = self.face_mesh.process(frame)

            face_mask = np.zeros((H, W), dtype=np.uint8)
            lip_mask = np.zeros((H, W), dtype=np.uint8)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]

                lip_points = self._points_from_landmarks(landmarks, self.LIPS_INDICES, W, H)
                face_points = self._points_from_landmarks(landmarks, self.FACE_OVAL_INDICES, W, H)

                lip_mask = self._build_region_mask(
                    lip_points,
                    H,
                    W,
                    dilate_size=dilate_lip,
                    close_size=close_lip,
                )
                face_mask = self._build_region_mask(
                    face_points,
                    H,
                    W,
                    dilate_size=dilate_face,
                    close_size=close_face,
                )
            elif self.reuse_previous_on_miss and prev_face_mask is not None and prev_lip_mask is not None:
                face_mask = prev_face_mask.copy()
                lip_mask = prev_lip_mask.copy()

            prev_face_mask = face_mask
            prev_lip_mask = lip_mask

            face_masks.append(face_mask)
            lip_masks.append(lip_mask)

        face_masks = torch.from_numpy(np.stack(face_masks)).float().to(self.device)
        lip_masks = torch.from_numpy(np.stack(lip_masks)).float().to(self.device)

        return face_masks, lip_masks

    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
