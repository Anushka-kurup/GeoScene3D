# 3D Spatial Anomaly Detection System

A complete pipeline for detecting anomalies in indoor scenes using RGB-D data, point cloud processing, and PointNet++ deep learning features.

## 📁 Project Structure

```
GeoScene3D/
├── preprocess_depth_to_pcd.py          # Core point cloud generation
├── pointnet_feature_extractor.py       # PointNet++ feature extraction
├── integrated_anomaly_detection.py     # Full pipeline (recommended)
├── batch_process_scenes.py             # Batch processing multiple scenes
├── dataset/
│   └── SUNRGBD/                        # Your dataset
└── output/                             # Generated results
```

## 🚀 Quick Start

### 1. Single Scene Analysis (Recommended)

Process one scene with the complete pipeline:

```bash
python integrated_anomaly_detection.py \
  /path/to/scene/folder \
  ./output/my_analysis \
  10
```

**Arguments:**
- `scene_folder`: Path to SUNRGBD scene
- `output_folder`: Where to save results (optional)
- `max_frames`: Number of frames to process (optional, default=10)

**Output:**
- Point clouds (.ply files)
- PointNet++ features (.npz files)
- Anomaly visualizations (colored point clouds)
- Analysis report (JSON)

### 2. Point Cloud Generation Only

If you only want to generate point clouds:

```bash
python preprocess_depth_to_pcd.py \
  /path/to/scene/folder \
  ./output \
  5
```

### 3. Batch Processing Multiple Scenes

Process entire dataset:

```bash
python batch_process_scenes.py \
  ./dataset/SUNRGBD \
  ./output/batch_results \
  20
```

**Arguments:**
- `dataset_root`: Root folder of SUNRGBD dataset
- `output_dir`: Output directory
- `max_scenes`: Max scenes to process (optional)

## 📊 What Each Script Does

### `preprocess_depth_to_pcd.py`
**Purpose:** Convert RGB-D images to 3D point clouds

**Features:**
- Loads RGB and depth images
- Reads camera intrinsics
- Generates point clouds using Open3D
- Compares consecutive frames
- Detects geometric anomalies (point-wise distance)
- Downsamples and cleans point clouds

**Key Functions:**
- `generate_point_cloud()` - Create point cloud from single frame
- `process_consecutive_frames()` - Process multiple frames
- `detect_temporal_anomaly()` - Compare two point clouds

### `pointnet_feature_extractor.py`
**Purpose:** Extract deep geometric features using PointNet++

**Features:**
- Implements PointNet++ architecture
- Hierarchical feature extraction (multi-scale)
- Scene comparison using feature similarity
- Can use pretrained weights (optional)

**Key Classes:**
- `PointNetPlusPlus` - Main network architecture
- `PointNetFeatureExtractor` - Easy-to-use wrapper
- `PointNetSetAbstraction` - Core PointNet++ layer

**Usage Example:**
```python
from pointnet_feature_extractor import PointNetFeatureExtractor

extractor = PointNetFeatureExtractor()
features = extractor.extract_features("scene.ply")
print(f"Global feature: {features['global_feature'].shape}")
```

### `integrated_anomaly_detection.py`
**Purpose:** Complete end-to-end pipeline

**Pipeline Steps:**
1. **Point Cloud Generation** - Convert RGB-D to 3D
2. **Feature Extraction** - Extract PointNet++ features
3. **Anomaly Detection** - Three methods:
   - Geometric (point distance)
   - Feature-based (similarity)
   - Combined (weighted)

**Output Report Includes:**
- Number of frames processed
- Geometric anomaly statistics
- Feature similarity scores
- Combined anomaly confidence
- Visualization files

### `batch_process_scenes.py`
**Purpose:** Process multiple scenes automatically

**Features:**
- Auto-discovers all valid scenes
- Progress tracking
- Error handling (continues on failure)
- JSON report with success/failure stats

## 📈 Understanding the Output

### Generated Files

**For each scene:**
```
output/scene_name/
├── frame_0000.ply                      # Raw point cloud
├── frame_0000_clean.ply                # Cleaned point cloud
├── features_0000.npz                   # PointNet++ features
├── geometric_anomaly_0000_to_0001.ply  # Anomaly visualization
├── processing_summary.json             # Frame processing info
└── integrated_analysis_report.json     # Complete analysis
```

### Anomaly Visualization

Point clouds are color-coded:
- **Gray points**: Normal/unchanged areas
- **Red points**: Detected anomalies (changes from previous frame)

Open `.ply` files with:
- Open3D viewer
- CloudCompare
- MeshLab
- Any 3D viewer

## 🎯 Anomaly Detection Methods

### 1. Geometric Anomaly Detection
- Compares point positions between frames
- Uses KD-tree for fast nearest neighbor search
- Threshold: 5cm distance
- **Good for:** Large movements, new objects

### 2. Feature-Based Anomaly Detection
- Compares PointNet++ global features
- Uses cosine similarity and L2 distance
- Threshold: 95% similarity
- **Good for:** Subtle changes, layout shifts

### 3. Combined Detection
- Weighted combination: 60% geometric + 40% feature
- Confidence score (0-100%)
- **Best for:** Robust detection with fewer false positives

## 🔧 Dependencies

```bash
pip install numpy
pip install open3d
pip install imageio
pip install scipy
pip install torch
pip install torchvision
```

## 💡 Use Cases

### Your Research Project
Based on your documents, this system addresses:

1. **3D Spatial Anomaly Detection**
   - Detect unauthorized objects
   - Identify layout changes
   - Work with occlusions and noise
   - Trigger alerts automatically

2. **Geometry-Grounded Vision-Language Models**
   - PointNet++ provides geometric features
   - Can be integrated with language models
   - Enables spatial reasoning queries

### Example Applications
- Security monitoring (detect intruders)
- Elderly care (fall detection, unusual behavior)
- Warehouse monitoring (inventory changes)
- Smart home (furniture moved, new items)

## 📝 Example Workflow

```bash
# 1. Test on single scene
python integrated_anomaly_detection.py \
  ./dataset/SUNRGBD/realsense/sa/2014_10_21-15_21_11-1311000073 \
  ./output/test_scene \
  5

# 2. Check results
ls ./output/test_scene/
cat ./output/test_scene/integrated_analysis_report.json

# 3. Batch process for training data
python batch_process_scenes.py \
  ./dataset/SUNRGBD \
  ./output/training_data \
  50

# 4. Analyze batch results
cat ./output/training_data/batch_report.json
```

## 🐛 Troubleshooting

### "0 points generated"
- Check depth image encoding (may need different scale factor)
- Verify intrinsics.txt format
- Try increasing `depth_trunc` parameter

### Out of Memory
- Reduce `max_frames`
- Decrease `num_points` in PointNet++ (default 2048)
- Process fewer scenes at once

### CUDA not available
- PointNet++ will automatically use CPU
- Slower but still works

## 🚀 Next Steps

1. **Add Semantic Segmentation**: Classify objects (chairs, tables, people)
2. **Integrate Language Model**: Query scenes with natural language
3. **Real-time Processing**: Optimize for live camera feeds
4. **Training**: Fine-tune PointNet++ on your specific scenes
5. **Alert System**: Add email/SMS notifications for anomalies

## 📚 References

- PointNet++: https://arxiv.org/abs/1706.02413
- SUNRGBD Dataset: https://rgbd.cs.princeton.edu/
- Open3D: http://www.open3d.org/

---

**Status:** ✅ All modules implemented and tested
**Last Updated:** January 2026