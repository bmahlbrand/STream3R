import os
import torch
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from stream3r.models.components.utils.load_fn import load_and_preprocess_images

from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from stream3r.models.components.utils.geometry import unproject_depth_map_to_point_map

# added imports for PyAV-based frame extraction
try:
    import av
except Exception as e:
    raise ImportError("PyAV is required to extract frames. Install with: pip install av") from e
from PIL import Image
import tempfile
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

model = STream3R.from_pretrained("yslan/STream3R").to(device)

# Use PyAV to load frames from the video file instead of reading from example_dir
video_path = "cod-playlist/179-youtube video #X5nomi2FzvU.mp4"
# Temporary directory to store extracted frames on disk (load_and_preprocess_images expects file paths)
tmp_dir = tempfile.mkdtemp(prefix="frames_")
image_paths = []

# extraction parameters
max_frames = 64  # limit number of frames to preprocess
target_fps = 1   # sample ~1 frame per second (adjust as needed)

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
# StreamSession supports KV cache management for both "causal" and "window" modes.
session = StreamSession(model, mode="causal")

with torch.no_grad():
    # Process images one by one to simulate streaming inference
    for i in range(images.shape[0]):
        image = images[i : i + 1]
        predictions = session.forward_stream(image)

        # Convert pose encoding to extrinsic and intrinsic matrices
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Convert tensors to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
        predictions['pose_enc_list'] = None # remove pose_enc_list

        # Generate world points from depth map
        print("Computing world points from depth map...")
        depth_map = predictions["depth"]  # (S, H, W, 1)
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points
        print(predictions.keys())
        # points_array = np.array(pts_list)    # [N, 3]
        # colors_array = np.array(cols_list)    # [N, 3]
        # pc = np.concatenate([points_array, colors_array], axis=1)
        # point_cloud = torch.from_numpy(pc).float()
        # ply_path = os.path.join("./", "points.ply")
        # print(f"Points passing confidence threshold: {conf_mask.sum()}")
        # print(f"Percentage of points kept: {100 * conf_mask.sum() / conf_mask.size:.2f}%")

        # print(f"Saving point cloud with {len(point_cloud)} points to {ply_path}")
        # trimesh.PointCloud(points_array, colors=colors_array).export(ply_path)

        # Clean up
        torch.cuda.empty_cache()
        
    session.clear()