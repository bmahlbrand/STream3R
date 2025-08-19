import os
import torch
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from stream3r.models.components.utils.load_fn import load_and_preprocess_images

from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from stream3r.models.components.utils.geometry import unproject_depth_map_to_point_map
from stream3r.utils.visual_utils import predictions_to_glb

# added imports for PyAV-based frame extraction
try:
    import av
except Exception as e:
    raise ImportError("PyAV is required to extract frames. Install with: pip install av") from e
from PIL import Image
import tempfile
import shutil

# ----------------------------
# GLB export configuration
# ----------------------------
# If None, we’ll default to "All" later (to match your request).
frame_filter = None
conf_thres = 50.0
mask_black_bg = False
mask_white_bg = False
show_cam = False
mask_sky = False
prediction_mode = "Predicted Pointmap"


device = "cuda" if torch.cuda.is_available() else "cpu"

model = STream3R.from_pretrained("yslan/STream3R").to(device)

# Use PyAV to load frames from the video file instead of reading from example_dir
video_path = "cod-playlist/179-youtube video #X5nomi2FzvU.mp4"
# Temporary directory to store extracted frames on disk (load_and_preprocess_images expects file paths)
tmp_dir = tempfile.mkdtemp(prefix="frames_")
image_paths = []

# extraction parameters
max_frames = 100  # limit number of frames to preprocess
target_fps = 1   # sample ~1 frame per second (adjust as needed)

print(f"Extracting frames from {video_path}... @ {target_fps} FPS")
try:
    container = av.open(video_path)
    stream = container.streams.video[0]
    # determine frame interval to approximate target_fps
    try:
        avg_rate = float(stream.average_rate) if stream.average_rate else None
    except Exception:
        avg_rate = None
    if avg_rate and avg_rate > 0:
        frame_interval = max(1, int(round(avg_rate / target_fps)))
    else:
        frame_interval = 1

    for i, frame in enumerate(container.decode(stream)):
        if i % frame_interval != 0:
            continue
        pil_img = frame.to_image()  # returns a PIL.Image
        frame_path = os.path.join(tmp_dir, f"frame_{i:06d}.png")
        pil_img.save(frame_path)
        image_paths.append(frame_path)
        if len(image_paths) >= max_frames:
            break

    if not image_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")

    images = load_and_preprocess_images(image_paths).to(device)
finally:
    # cleanup extracted frames
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

print(f"Extracted {len(images)} frames from {video_path}")
print(f"Processing frames...")

mode = "causal"
# StreamSession supports KV cache management for both "causal" and "window" modes.
session = StreamSession(model, mode="causal")
# Where to drop GLB outputs (by default: alongside the video, in a subdir)
video_stem = os.path.splitext(os.path.basename(video_path))[0]
target_dir = os.path.join(os.path.dirname(video_path), f"{video_stem}_outputs")
os.makedirs(target_dir, exist_ok=True)

with torch.no_grad():
    # Process images one by one to simulate streaming inference
    for i in range(images.shape[0]):
        image = images[i : i + 1].to(device)
        outputs = session.forward_stream(image)
        predictions = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                # detach and clone to produce an independent CPU tensor
                predictions[k] = v.detach().cpu().clone()
            else:
                predictions[k] = v
        # # Convert pose encoding to extrinsic and intrinsic matrices
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Convert tensors to numpy
        for key in list(predictions.keys()):
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
        predictions["pose_enc_list"] = None  # remove pose_enc_list

        # Generate world points from depth map
        print("Computing world points from depth map...")
        depth_map = predictions["depth"]  # (S, H, W, 1)
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points
        # print(predictions.keys())

        # Clean up
        torch.cuda.empty_cache()

        # ----------------------------
    # GLB export (requested block)
    # ----------------------------
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu()

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}"
        f"_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}"
        f"_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}_mode{mode}.glb",
    )
    predictions["images"] = images.cpu().numpy()

    # Convert predictions to GLB
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

    print(".")

    session.clear()
