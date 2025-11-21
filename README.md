# CADRNet: Cognitively-Inspired Active Vision for 3D Reasoning Segmentation


## 📄 About CADRNet

CADRNet (Cognitively-Inspired Active Vision via Differentiable Rendering Network) is a novel approach for 3D reasoning segmentation that draws inspiration from human active vision theory. Unlike traditional methods that directly adapt Multimodal Large Language Models (MLLMs) to 3D point clouds, CADRNet simulates the human cognitive process of actively selecting informative 2D views to understand 3D scenes.

### Core Innovation
CADRNet addresses two fundamental challenges in 3D reasoning segmentation:
- **Erroneous Localization**: Traditional MLLMs trained on text-image pairs struggle with the unstructured nature of 3D point clouds, leading to ambiguous object localization
- **Boundary Ambiguity**: Direct adaptation of MLLMs to 3D data results in coarse object boundary delineation

### Technical Approach
CADRNet comprises two key modules:
1. **Viewpoint Position Adaptive Learning (VPAL)**: Optimizes camera parameters via gradient descent on segmentation loss, enabling end-to-end rendering of task-relevant 2D views
2. **Semantic Coherence Grouping Fusion**: Fuses textural details from rendered views with precise geometric structures from point clouds for holistic 3D-aware perception

### Performance
CADRNet achieves state-of-the-art results on multiple benchmarks:
- **Multi-Scan Dataset**: 3.43%↑ gIoU improvement in multi-object reasoning segmentation
- **Instruct3D Dataset**: 2.31%↑ mIoU improvement in single-object reasoning segmentation
- **3D Visual Grounding**: Competitive results on standard benchmarks

## 🎯 Multi-Scan Dataset

Multi-Scan is the first large-scale multi-object reasoning segmentation dataset introduced in our CADRNet paper. Unlike traditional single-object segmentation tasks, Multi-Scan focuses on **multi-object instance reasoning**, where models must simultaneously understand, reason about, and segment multiple objects within complex 3D scenes based on natural language queries.

**Key Features**:
- Multi-object reasoning with 1-3 target objects per query
- Complex spatial relationships and functional reasoning
- Built upon ScanNet with enhanced reasoning annotations
- Supports both single-object and multi-object evaluation

## 📅 Open-Source Schedule

We are releasing our work in three strategic phases to ensure quality, reproducibility, and community engagement:

### 🔬 Stage 1: Multi-Scan DataLoader (Current Release) ✅
**Status**: Released  
**Content**:
- PyTorch DataLoader for Multi-Scan dataset
- Multi-Scan **Train & Val splits** with 5,391 multi-object reasoning samples
- Data preprocessing and batch collation utilities
- Documentation and usage examples

**Note**: This represents the first phase of our open-source initiative, providing immediate access to our novel multi-object reasoning dataset.

### 📊 Stage 2: Complete Multi-Scan & Multi-Scan++ (Coming Soon) 🔄
**Status**: In preparation (Expected: Early 2025)  
**Content**:
- Multi-Scan **Test set** with complete annotations
- Full Multi-Scan dataset with extended reasoning annotations
- Multi-Scan++ with additional challenging scenarios and complex object relationships
- Enhanced evaluation metrics and benchmarking protocols
- Comprehensive dataset analysis and statistics

### ⚙️ Stage 3: CADRNet Training & Evaluation Framework (Future Release) 🔜
**Status**: Planned (Expected: Mid 2025)  
**Content**:
- Complete CADRNet model implementation with VPAL and fusion modules
- Training scripts with configurations for different dataset scales
- Evaluation protocols and comprehensive metrics
- Pre-trained model weights for multiple benchmarks
- Inference pipeline and visualization tools
- Reproducibility guidelines and experiment configurations

**Stay updated! ⭐ Star this repository to follow our progress and receive notifications for each release phase.**

## 📁 Directory Structure

```
project_root/
├── Multi-scan-dataloader/
│   ├── Multi-Scan_dataset/                # Multi-Scan JSON data files (Train + Val)
│   │   ├── Multi-Scan-1-v2.json          # 1-object reasoning data (392 samples)
│   │   ├── Multi-Scan-2-v2.json          # 2-object reasoning data (2,499 samples)
│   │   └── Multi-Scan-3-v2.json          # 3-object reasoning data (2,500 samples)
│   ├── multi_scan_dataloader.py          # Main DataLoader implementation
│   ├── README.md                         # This documentation
│   └── requirements.txt                  # Python dependencies
│
└── scans/                                # ScanNet dataset (required)
    ├── scene0000_00/
    │   ├── scene0000_00_vh_clean_2.labels.ply          # Point cloud with labels
    │   ├── scene0000_00_vh_clean_2.0.010000.segs.json  # Segmentation indices
    │   └── scene0000_00.aggregation.json               # Object aggregation
    ├── scene0000_01/
    │   └── ...
    └── ...
```

## 🔧 Environment Requirements

### Python Version
- Python 3.8 or higher

### Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.0.1+cu118 | Deep learning framework |
| numpy | 1.24.3 | Numerical operations |
| plyfile | 0.9 | PLY file format handling |

### Optional Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| torchvision | 0.15.2+cu118 | Vision utilities |
| open3d | 0.17.0 | Point cloud visualization |
| pandas | 2.0.3 | Data analysis |
| tqdm | 4.66.1 | Progress tracking |

## 📦 Data Requirements

### ScanNet Dataset
Multi-Scan builds upon the ScanNet dataset. Ensure you have:

**Required Files per Scene**:
- `*_vh_clean_2.labels.ply`: 3D point cloud with vertex positions (x,y,z), RGB colors, and semantic labels
- `*_vh_clean_2.0.010000.segs.json`: Segmentation indices mapping points to segment IDs
- `*.aggregation.json`: Object aggregation mapping segment IDs to object IDs with class labels

**Note**: Download ScanNet and ensure all three file types are available for scenes referenced in Multi-Scan JSON files.

## 🚀 Quick Start

### Basic Usage

```python
from multi_scan_dataloader import MultiScanDataset, collate_multi_scan
from torch.utils.data import DataLoader

# Initialize dataset
dataset = MultiScanDataset(
    reasoning_dir="path/to/Multi-Scan_dataset",
    scans_dir="path/to/scans",
    load_point_cloud=True,
    training=True  # True for training/validation, False for testing
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_multi_scan
)

# Training loop
for batch in dataloader:
    coords = batch['coords']              # [B*N, 4] - point coordinates
    class_labels = batch['class_labels']  # List[List[str]] - object classes
    gt_pmasks = batch['gt_pmasks']        # List[List[Tensor]] - ground truth masks
    # Your training code here
```

### Test Run

```bash
# Windows
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python multi_scan_dataloader.py --reasoning_dir "Multi-Scan_dataset" --scans_dir "path/to/scans"

# Linux/Mac
export KMP_DUPLICATE_LIB_OK=TRUE
python multi_scan_dataloader.py --reasoning_dir "Multi-Scan_dataset" --scans_dir "path/to/scans"
```

## 📊 Dataset Statistics

**Current Release (Stage 1 - Train + Val)**:
- **Total Samples**: 5,391 multi-object reasoning queries
- **Distribution by Object Count**:
  - 1-object queries: 392 samples
  - 2-object queries: 2,499 samples  
  - 3-object queries: 2,500 samples
- **Point Cloud Size**: 50K-500K points per scene (full resolution)
- **Object Instances**: 1-10+ objects per sample
- **Semantic Categories**: 1-5+ unique classes per sample
- **Query Types**: Functional reasoning, spatial relationships, attribute-based queries

**Upcoming (Stage 2)**:
- Complete test set with held-out scenes
- Multi-Scan++ with complex reasoning scenarios
- Extended evaluation metrics and benchmarks

