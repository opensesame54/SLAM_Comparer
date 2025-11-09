# RGB-D SLAM with SIFT Features

A complete RGB-D SLAM (Simultaneous Localization and Mapping) implementation using SIFT features, designed for TUM RGB-D datasets.

## Features

- ✅ **Feature-based tracking** using SIFT descriptors
- ✅ **Motion estimation** with 3D-3D correspondences and RANSAC
- ✅ **Keyframe-based mapping** for efficient reconstruction
- ✅ **Loop closure detection** to reduce drift
- ✅ **Dense/Sparse point cloud generation**
- ✅ **Post-processing pipeline** (outlier removal, floor filling, meshing)
- ✅ **Trajectory visualization**

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Git
git --version
```

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/opensesame54/SLAM_Comparer.git
cd SLAM_Comparer

# 2. Switch to Python SLAM branch
git checkout python-rgbd-slam

# 3. Navigate to the implementation
cd slam_py/rgbd_slam_python

# 4. Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Download a TUM RGB-D dataset:

```bash
# Option 1: Using wget (Linux/Mac)
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
tar -xvzf rgbd_dataset_freiburg1_room.tgz

# Option 2: Using PowerShell (Windows)
Invoke-WebRequest -Uri "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz" -OutFile "dataset.tgz"
tar -xvzf dataset.tgz

# Option 3: Manual download
# Visit: https://vision.in.tum.de/data/datasets/rgbd-dataset/download
# Download "freiburg1_room" dataset
# Extract to rgbd_slam_python/ folder
```

Your folder structure should look like:
```
rgbd_slam_python/
├── main.py
├── requirements.txt
├── rgbd_dataset_freiburg1_room/  # ← Dataset here
│   ├── rgb/
│   ├── depth/
│   └── ...
└── ...
```

## Usage

### 1. Basic SLAM (Sparse Reconstruction)

```bash
# Process entire dataset with sparse point cloud
python main.py --dataset rgbd_dataset_freiburg1_room

# Process limited frames (faster testing)
python main.py --dataset rgbd_dataset_freiburg1_room --max_frames 100

# Skip frames for faster preview
python main.py --dataset rgbd_dataset_freiburg1_room --skip_frames 3
```

**Output:** `output/point_clouds/reconstruction.ply`

### 2. Dense Reconstruction

```bash
# Dense point cloud (slower but more complete)
python main.py --dataset rgbd_dataset_freiburg1_room --dense

# Dense with custom config
python main.py --dataset rgbd_dataset_freiburg1_room --config config/config_dense.yaml
```

**Output:** `output/point_clouds/reconstruction.ply` (higher density)

### 3. Dense + Post-Processing (Recommended)

```bash
# Full pipeline: SLAM + outlier removal + floor filling + meshing
python main.py --dataset rgbd_dataset_freiburg1_room --dense --postprocess --visualize
```

**Output:** 
- `output/point_clouds/reconstruction.ply` (raw)
- `output/point_clouds/reconstruction_processed.ply` (cleaned & filled)
- `output/point_clouds/reconstruction_mesh.ply` (mesh)

### 4. Post-Processing Only (Existing Reconstruction)

If you already have a reconstruction:

```bash
# Fast post-processing (recommended for large clouds)
python postprocess_reconstruction.py \
    --input output/point_clouds/reconstruction.ply \
    --output output/point_clouds/final.ply \
    --downsample-first 0.01 \
    --mesh-depth 8 \
    --sample-points 500000 \
    --visualize

# High-quality (slower)
python postprocess_reconstruction.py \
    --input output/point_clouds/reconstruction.ply \
    --output output/point_clouds/final_hq.ply \
    --downsample-first 0.01 \
    --mesh-depth 9 \
    --sample-points 1000000 \
    --visualize

# Skip specific steps
python postprocess_reconstruction.py \
    --input output/point_clouds/reconstruction.ply \
    --no-outliers \
    --no-floor \
    --visualize
```

## Configuration

### config/config.yaml (Default - Sparse)

```yaml
camera:
  fx: 525.0
  fy: 525.0
  cx: 319.5
  cy: 239.5
  width: 640
  height: 480
  depth_scale: 5000.0

sift:
  n_features: 2000
  contrast_threshold: 0.04
  edge_threshold: 10
  sigma: 1.6

matcher:
  ratio_threshold: 0.75
  method: 'FLANN'

min_matches: 20
keyframe_interval: 5
max_tracking_lost: 10
use_dense_cloud: false
dense_cloud_step: 5
use_loop_closure: true
loop_min_matches: 50
```

### config/config_dense.yaml (Dense Reconstruction)

```yaml
camera:
  fx: 525.0
  fy: 525.0
  cx: 319.5
  cy: 239.5
  width: 640
  height: 480
  depth_scale: 5000.0

sift:
  n_features: 2500
  contrast_threshold: 0.03
  edge_threshold: 10
  sigma: 1.6

matcher:
  ratio_threshold: 0.75
  method: 'FLANN'

min_matches: 25
keyframe_interval: 3
max_tracking_lost: 10
use_dense_cloud: true
dense_cloud_step: 2
use_loop_closure: true
loop_min_matches: 50
```

## Command-Line Options

### main.py

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `rgbd_dataset_freiburg1_room` | Path to TUM dataset |
| `--config` | `config/config.yaml` | Configuration file |
| `--output` | `output/point_clouds/reconstruction.ply` | Output file |
| `--max_frames` | `None` | Limit number of frames |
| `--start_frame` | `0` | Starting frame index |
| `--skip_frames` | `1` | Process every Nth frame |
| `--dense` | `False` | Enable dense reconstruction |
| `--postprocess` | `False` | Run post-processing |
| `--visualize` | `False` | Show visualization |
| `--verbose` | `False` | Detailed frame info |

### postprocess_reconstruction.py

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `output/point_clouds/reconstruction.ply` | Input PLY file |
| `--output` | `output/point_clouds/reconstruction_processed.ply` | Output file |
| `--downsample-first` | `0.01` | Downsample voxel size (0=skip) |
| `--mesh-depth` | `8` | Poisson mesh depth (8-10) |
| `--sample-points` | `500000` | Points to sample from mesh |
| `--no-outliers` | `False` | Skip outlier removal |
| `--no-floor` | `False` | Skip floor filling |
| `--no-mesh` | `False` | Skip meshing |
| `--visualize` | `False` | Show result |

## Example Workflows

### Workflow 1: Quick Preview (1-2 minutes)

```bash
# Sparse reconstruction, every 5th frame
python main.py --dataset rgbd_dataset_freiburg1_room --skip_frames 5 --max_frames 200
```

### Workflow 2: Standard Reconstruction (5-10 minutes)

```bash
# Sparse reconstruction, all frames
python main.py --dataset rgbd_dataset_freiburg1_room
```

### Workflow 3: High-Quality Dense (30-60 minutes)

```bash
# Dense reconstruction with post-processing
python main.py --dataset rgbd_dataset_freiburg1_room --dense --postprocess

# Additional high-quality post-processing
python postprocess_reconstruction.py \
    --input output/point_clouds/reconstruction_processed.ply \
    --output output/point_clouds/final_hq.ply \
    --mesh-depth 9 \
    --sample-points 1000000 \
    --visualize
```

### Workflow 4: Large Dataset (>1M points)

```bash
# Run SLAM without post-processing
python main.py --dataset rgbd_dataset_freiburg1_room --dense

# Post-process with downsampling (much faster)
python postprocess_reconstruction.py \
    --input output/point_clouds/reconstruction.ply \
    --output output/point_clouds/final.ply \
    --downsample-first 0.02 \
    --mesh-depth 8 \
    --sample-points 500000 \
    --visualize
```

## Viewing Results

### CloudCompare (Recommended)

1. Download: https://www.cloudcompare.org/
2. Open: `File > Open > reconstruction.ply`
3. Navigate: Mouse wheel (zoom), Middle-click (rotate), Shift+Middle-click (pan)

### MeshLab

1. Download: https://www.meshlab.net/
2. Open: `File > Import Mesh > reconstruction.ply`

### Open3D (Python)

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("output/point_clouds/reconstruction.ply")
o3d.visualization.draw_geometries([pcd])
```

## Output Files

After running SLAM:

```
output/point_clouds/
├── reconstruction.ply              # Raw point cloud
├── reconstruction_processed.ply    # Cleaned + filled (if --postprocess)
├── reconstruction_mesh.ply         # Mesh (if --postprocess)
├── trajectory.txt                  # Camera poses (TUM format)
└── trajectory.png                  # Trajectory plot
```

## Troubleshooting

### Issue: "No module named 'cv2'"

```bash
pip install opencv-contrib-python
```

### Issue: "No module named 'open3d'"

```bash
pip install open3d
```

### Issue: "Dataset not found"

Make sure dataset is in the correct location:
```bash
ls rgbd_dataset_freiburg1_room/rgb/
# Should show many .png files
```

### Issue: "0 keyframes, 0 points"

Check depth images:
```python
import cv2
depth = cv2.imread('rgbd_dataset_freiburg1_room/depth/1305031102.160407.png', -1)
print(f"Depth range: {depth.min()} - {depth.max()}")
# Should show: Depth range: 0 - ~5000
```

### Issue: Post-processing stuck/slow

For large point clouds (>10M points), use downsampling:
```bash
python postprocess_reconstruction.py \
    --input output/point_clouds/reconstruction.ply \
    --downsample-first 0.02 \
    --no-outliers
```

## Performance Tips

1. **Fast Preview:** Use `--skip_frames 5 --max_frames 200`
2. **Speed vs Quality:** Lower `dense_cloud_step` = slower but denser
3. **Large Datasets:** Use `--downsample-first 0.02` in post-processing
4. **Memory Issues:** Process in chunks with `--start_frame` and `--max_frames`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SLAM Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│  1. Feature Detection (SIFT)                                │
│  2. Feature Matching (FLANN)                                │
│  3. Motion Estimation (3D-3D RANSAC)                        │
│  4. Keyframe Selection                                       │
│  5. Point Cloud Generation (Sparse/Dense)                   │
│  6. Loop Closure Detection                                   │
│  7. Trajectory Optimization (optional)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Post-Processing Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│  1. Statistical Outlier Removal                             │
│  2. Floor Plane Detection & Filling                         │
│  3. Poisson Surface Reconstruction (Meshing)                │
│  4. Dense Point Sampling from Mesh                          │
└─────────────────────────────────────────────────────────────┘
```

## Dependencies

- Python 3.8+
- OpenCV (opencv-contrib-python)
- NumPy
- PyYAML
- Matplotlib
- Open3D
- SciPy

## Datasets

This implementation is tested with TUM RGB-D datasets:
- https://vision.in.tum.de/data/datasets/rgbd-dataset


