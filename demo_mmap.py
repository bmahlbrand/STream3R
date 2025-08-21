import json
import traceback
from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict

from tqdm import tqdm

import torch
import torch.nn.functional as F

# new imports from demo that we will use to run the model and export GLB
import os
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from stream3r.models.components.utils.geometry import unproject_depth_map_to_point_map
from stream3r.utils.visual_utils import predictions_to_glb

class ChunkedVideo:
    """Represents a logical video backed by multiple chunk tensor files.

    Each chunk file: torch.save of uint8 tensor [n, C=3, H, W]. We lazily load
    only the needed chunks when assembling windows.
    """
    def __init__(self, splits_dir: Path, cache_size: int = 2, persist_index: bool = False, allow_skip_corrupt: bool = False):
        self.splits_dir = splits_dir
        # A stable identifier (relative path string) may be used in global index
        self.video_id = splits_dir.name
        self.cache_size = cache_size
        self.persist_index = persist_index
        self.allow_skip_corrupt = allow_skip_corrupt
        self.chunk_files = sorted([p for p in splits_dir.glob("*_rgb.pt") if p.is_file()])
        if not self.chunk_files:
            raise FileNotFoundError(f"No *_rgb.pt chunk files in {splits_dir}")
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._chunk_sizes: List[int] = []
        self._shape: Optional[Tuple[int, int, int]] = None  # (C,H,W)
        self._cum_sizes: List[int] = []  # exclusive cum sum
        self._meta_path = splits_dir / "index_meta.json"
        self._build_index()

    def _build_index(self):
        if self.persist_index and self._meta_path.exists():
            try:
                meta = json.loads(self._meta_path.read_text())
                if meta.get("files") == [str(p.name) for p in self.chunk_files]:
                    self._chunk_sizes = meta["sizes"]
                    self._shape = tuple(meta["shape"])  # type: ignore
                    total = 0
                    self._cum_sizes = []
                    for s in self._chunk_sizes:
                        self._cum_sizes.append(total)
                        total += s
                    return
            except Exception:
                pass  # fall back to recompute
        # Need to scan files
        self._chunk_sizes.clear()
        self._cum_sizes.clear()
        total = 0
        corrupt_files = []
        valid_files = []
        for cf in tqdm(self.chunk_files, desc="Loading chunks", unit="chunk"):
            try:
                t = torch.load(cf, map_location="cpu", weights_only=True)  # [n,3,H,W]
            except Exception as e:
                corrupt_files.append((cf, str(e)))
                if not self.allow_skip_corrupt:
                    raise RuntimeError(f"Corrupted chunk file: {cf} error={e}") from e
                continue
            # Basic structural validation
            if not isinstance(t, torch.Tensor) or t.ndim != 4 or t.shape[1] != 3:
                msg = f"Invalid tensor shape in {cf}: expected [N,3,H,W] got {getattr(t,'shape',None)}"
                corrupt_files.append((cf, msg))
                if not self.allow_skip_corrupt:
                    raise RuntimeError(msg)
                continue
            n, c, h, w = t.shape
            if self._shape is None:
                self._shape = (c, h, w)
            else:
                if self._shape != (c, h, w):
                    msg = f"Inconsistent chunk frame shape in {cf}: {c,h,w} vs {self._shape}"
                    if not self.allow_skip_corrupt:
                        raise ValueError(msg)
                    corrupt_files.append((cf, msg))
                    continue
            self._chunk_sizes.append(n)
            self._cum_sizes.append(total)
            total += n
            valid_files.append(cf)
        # Replace chunk_files with only valid ones if skipping corrupt
        if self.allow_skip_corrupt:
            self.chunk_files = valid_files
            if corrupt_files:
                print(f"[ChunkedVideo] {self.splits_dir}: skipped {len(corrupt_files)} corrupt chunks; kept {len(valid_files)}")
                for cf, err in corrupt_files[:5]:
                    print(f"  corrupt: {cf.name} -> {err}")
                if len(corrupt_files) > 5:
                    print(f"  ... {len(corrupt_files)-5} more corrupt files")
        if not self._chunk_sizes:
            raise RuntimeError(f"No valid chunk files remain in {self.splits_dir} (corrupt or incompatible)")
        if self.persist_index:
            try:
                meta = {
                    "files": [p.name for p in self.chunk_files],
                    "sizes": self._chunk_sizes,
                    "shape": list(self._shape),
                }
                self._meta_path.write_text(json.dumps(meta))
            except Exception:
                pass

    @property
    def num_frames(self) -> int:
        return sum(self._chunk_sizes)

    @property
    def frame_shape(self) -> Tuple[int, int, int]:  # (C,H,W)
        if self._shape is None:
            raise RuntimeError("Frame shape not initialized")
        return self._shape

    def _load_chunk(self, idx: int) -> torch.Tensor:
        cf = self.chunk_files[idx]
        key = str(cf)
        if key in self._cache:
            val = self._cache.pop(key)
            self._cache[key] = val  # move to end (recent)
            return val
        t = torch.load(cf, map_location="cpu", weights_only=True)  # uint8
        self._cache[key] = t
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # evict LRU
        return t

    def _global_to_chunk(self, frame_idx: int) -> Tuple[int, int]:
        # binary search over cum sizes
        # cum[i] = start index of chunk i
        lo, hi = 0, len(self._chunk_sizes) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self._cum_sizes[mid]
            end = start + self._chunk_sizes[mid]
            if frame_idx < start:
                hi = mid - 1
            elif frame_idx >= end:
                lo = mid + 1
            else:
                return mid, frame_idx - start
        raise IndexError(frame_idx)

    def get_frames(self, indices: List[int]) -> torch.Tensor:
        """Return frames stacked [T,3,H,W] as float in [0,1]."""
        # Group by chunk
        by_chunk: Dict[int, List[Tuple[int, int]]] = {}
        for pos, gidx in enumerate(indices):
            c_idx, off = self._global_to_chunk(gidx)
            by_chunk.setdefault(c_idx, []).append((pos, off))
        C, H, W = self._shape
        out = torch.empty(len(indices), C, H, W, dtype=torch.float32)
        for c_idx, lst in by_chunk.items():
            ck = self._load_chunk(c_idx)  # [n,3,H,W] uint8
            for pos, off in lst:
                out[pos] = ck[off].float() / 255.0
        return out

# def square_and_resize(images: torch.Tensor, load_res: int) -> torch.Tensor:
#     """Resize and pad images to square of size (load_res, load_res)."""
#     square_images = []
#     for img in images:
#         c, h, w = img.shape
#         # Scale to keep aspect ratio
#         if h >= w:
#             new_h = load_res
#             new_w = max(1, int(w * load_res / h))
#         else:
#             new_w = load_res
#             new_h = max(1, int(h * load_res / w))
            
#         # Make dimensions divisible by 14
#         new_h = (new_h // 14) * 14
#         new_w = (new_w // 14) * 14
        
#         img_resized = F.interpolate(
#             img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
#         ).squeeze(0)
        
#         # Pad to square
#         pad_top = (load_res - new_h) // 2
#         pad_bottom = load_res - new_h - pad_top
#         pad_left = (load_res - new_w) // 2
#         pad_right = load_res - new_w - pad_left
        
#         img_padded = F.pad(
#             img_resized,
#             (pad_left, pad_right, pad_top, pad_bottom),
#             mode="constant",
#             value=1.0,
#         )
#         square_images.append(img_padded)
#     return torch.stack(square_images)
def resize_keep_aspect_set_width(x: torch.Tensor, target_w: int = 518) -> torch.Tensor:
    """
    x: [C,H,W] or [B,C,H,W] float/uint8 tensor
    Returns tensor resized so width = target_w, height scaled to keep aspect.
    """
    single = (x.dim() == 3)
    if single:
        x_in = x.unsqueeze(0)          # [1,C,H,W]
    else:
        x_in = x
    _, _, H, W = x_in.shape
    new_h = round(H * target_w / W)
    new_h = (new_h // 14) * 14  # or +14 if you prefer rounding up
    resized = F.interpolate(
        x_in, size=(new_h, target_w), mode="bilinear", align_corners=False, antialias=True
    )
    return resized.squeeze(0) if single else resized

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run STream3R on a ChunkedVideo (mmap .pt chunks).")
    parser.add_argument("splits_dir", help="Directory containing *_rgb.pt chunk files")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum number of frames to process")
    parser.add_argument("--sample-rate", type=int, default=1, help="Take every Nth frame")
    parser.add_argument("--cache-size", type=int, default=2, help="Chunk cache size (number of chunk tensors to keep)")
    parser.add_argument("--out-dir", default=None, help="Where to write GLB outputs (defaults next to splits_dir)")
    args = parser.parse_args()

    splits_path = Path(args.splits_dir)
    if not splits_path.exists() or not splits_path.is_dir():
        raise SystemExit(f"Invalid splits_dir: {splits_path}")

    # instantiate ChunkedVideo
    cv = ChunkedVideo(splits_path, cache_size=args.cache_size)

    device = args.device
    model = STream3R.from_pretrained("yslan/STream3R").to(device)

    mode = "causal"
    session = StreamSession(model, mode=mode)

    total_frames = cv.num_frames
    indices = list(range(0, total_frames, args.sample_rate))[: args.max_frames]
    if not indices:
        raise SystemExit("No frames selected from ChunkedVideo")

    images = resize_keep_aspect_set_width(cv.get_frames(indices).to(device))  # [T,3,H,W] float32 in [0,1]
    print(f"Loaded {images.shape[0]} frames from {splits_path} (device={device})")

    # run streaming inference frame-by-frame to simulate original demo behavior
    target_dir = args.out_dir or str(splits_path.parent / f"{splits_path.name}_outputs")
    os.makedirs(target_dir, exist_ok=True)

    predictions = {}
    with torch.no_grad():
        for i in range(images.shape[0]):
            image = images[i : i + 1].to(device)
            outputs = session.forward_stream(image)
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    predictions[k] = v.detach().cpu().clone()
                else:
                    predictions[k] = v
            # compute extrinsic/intrinsic using image shape (H,W)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic

            # convert to numpy where needed and compute world points from depth
            for key in list(predictions.keys()):
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)
            predictions["pose_enc_list"] = None

            depth_map = predictions["depth"]
            world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
            predictions["world_points_from_depth"] = world_points

            torch.cuda.empty_cache()

        # After loop, prepare and export GLB (similar to demo.py)
        for key in list(predictions.keys()):
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu()

        frame_filter = "All"
        conf_thres = 50.0
        mask_black_bg = False
        mask_white_bg = False
        show_cam = False
        mask_sky = False
        prediction_mode = "Predicted Pointmap"

        video_stem = splits_path.name
        glbfile = os.path.join(
            target_dir,
            f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}"
            f"_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}"
            f"_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}_mode{mode}.glb",
        )

        # attach the original preprocessed images (CPU numpy) for visualization/export
        predictions["images"] = images.cpu().numpy()
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)
        print(f"Exported GLB to {glbfile}")

    session.clear()