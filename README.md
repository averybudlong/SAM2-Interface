# SAM2-Interface

### OVERVIEW:
- Gradio frontend to allow selection of ROI's
- Modified SAM2 to reduce VRAM usage and allow processing of long form videos (https://github.com/facebookresearch/sam2/issues/264#issuecomment-2310740437)
- Use HDF5 file to store mask data for every frame in the video
