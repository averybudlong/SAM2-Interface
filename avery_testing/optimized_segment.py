import os
import torch
from sam2.build_sam import build_sam2_video_predictor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import perf_counter
import cv2
import io

# ffmpeg -i "Z:\IndividualStudies\1. Mother Child Dynamics (F32) study\Data\3. ET Data\Child Raw Files\571_Child\Speech Task\exports\000\571_Child_Speech_Corrected.mp4" -vf "select=between(n\,3897\,5396),setpts=PTS-STARTPTS" -q:v 2 -start_number 0 C:\Users\abudlong\Desktop\frames_571/'%05d.jpg'

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"using device: {device}")
else:
    device = torch.device("cpu")
    print(f"using device: {device} NOT GPU!!!")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def add_clicks(predictor, inference_state, ann_frame_idx, ann_obj_id, points, labels):
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
    )

    return out_obj_ids, out_mask_logits

# Function to convert matplotlib figure to OpenCV compatible image
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "../notebooks/videos/585"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Get first frame to determine video dimensions
first_frame = Image.open(os.path.join(video_dir, frame_names[0]))
width, height = first_frame.size

# Initialize video writer
output_path = './output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
fps = 29.93  # frames per second
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=True)

prompts = {}  # hold all the clicks we add for visualization

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
# points containes coordinates of clicks
points = np.array([[1050, 390], [1050, 600]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
prompts[ann_obj_id] = points, labels
add_clicks(predictor, inference_state, ann_frame_idx, ann_obj_id, points, labels)

ann_frame_idx = 0
ann_obj_id = 2
points = np.array([[145, 390]], dtype=np.float32)
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
add_clicks(predictor, inference_state, ann_frame_idx, ann_obj_id, points, labels)

ann_frame_idx = 270
ann_obj_id = 3
points = np.array([[500, 800],[900, 800]], dtype=np.float32)
labels = np.array([1, 1], np.int32)
prompts[ann_obj_id] = points, labels
add_clicks(predictor, inference_state, ann_frame_idx, ann_obj_id, points, labels)

ann_frame_idx = 390
ann_obj_id = 4
points = np.array([[50, 500]], dtype=np.float32)
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
add_clicks(predictor, inference_state, ann_frame_idx, ann_obj_id, points, labels)

# Initialize for visualization
vis_frame_stride = 1
plt.close("all")

# Counter to keep track of processed frames
frame_counter = 0
tprev = -1

# Create output directory if it doesn't exist
os.makedirs("./video_frames", exist_ok=True)

# Process frames as they're propagated
print("Starting video generation...")
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    if (perf_counter() > tprev + 1.0) and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print("VRAM:", (total_bytes - free_bytes) // 1_000_000, "MB")
        tprev = perf_counter()
    
    # Process only every vis_frame_stride frames
    if frame_counter % vis_frame_stride == 0:
        # Create a new figure with no border/axes
        fig = plt.figure(figsize=(width/100, height/100), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Load the original frame
        frame = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        ax.imshow(frame)
        
        # Process masks for this frame directly without storing them
        for i, out_obj_id in enumerate(out_obj_ids):
            # Convert mask logits to binary mask directly
            out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            show_mask(out_mask, ax, obj_id=out_obj_id)
        
        # Convert the figure to an OpenCV compatible image
        img = fig2img(fig)
        
        # OpenCV uses BGR format, matplotlib uses RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        video_writer.write(img_bgr)
        
        # Optional: Save individual frames if needed for debugging
        # cv2.imwrite(f"./video_frames/{out_frame_idx:04d}.jpg", img_bgr)
        
        # Print progress
        if out_frame_idx % 10 == 0:
            print(f"Processed frame {out_frame_idx}")
            
        # Close the figure to free memory
        plt.close(fig)
    
    frame_counter += 1

# Release video writer
video_writer.release()
print(f"Video generation complete. Output saved to {output_path}")