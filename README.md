# SAM2-Interface

### OVERVIEW:
- We need a way to easily select points inside each ROI to be used as input into SAM
  - There might be an existing UI or we could build a simple one with a framework like Gradio
  - Similar to the demo (https://sam2.metademolab.com/demo) 
  - Output should be a list of: frame #, (x, y) coordinate, ROI ID
- Modify SAM2 to reduce VRAM usage and allow processing of long form videos
  - Seems like a common issue and there are some good ideas here: (https://github.com/facebookresearch/sam2/issues/264#issuecomment-2310740437)
- Use HDF5 file to store mask data for every frame in the video
  - 
