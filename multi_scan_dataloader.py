"""
Multi-Scan Dataset for Multi-Object Instance Reasoning Segmentation
Loads multi-object samples with individual masks for each object

Usage:
    python multi_scan_dataloader.py --reasoning_dir "your_data_path/Multi-Scan" --scans_dir "your_data_path/scans"

Required Input:
    --reasoning_dir: Path to Multi-Scan JSON files directory
                    Should contain: Multi-Scan-1-v2.json, Multi-Scan-2-v2.json, Multi-Scan-3-v2.json
    --scans_dir: Path to ScanNet scans directory
                Should contain scene folders with:
                - scene_id_vh_clean_2.labels.ply
                - scene_id_vh_clean_2.0.010000.segs.json
                - scene_id.aggregation.json

Output Format (Single Sample):
    {
        'ann_ids': int,                         # Sample index
        'scan_ids': str,                        # Scene ID (e.g., 'scene0276_00')
        'coord': Tensor[N, 3],                  # Integer coordinates
        'coord_float': Tensor[N, 3],            # Float coordinates
        'feat': Tensor[N, 3],                   # RGB features
        'superpoint': Tensor[N],                # Segment indices
        'object_ids': List[int],                # Object IDs (e.g., [0, 1, 2, 3, 4, 26, 16])
        'object_names': List[str],              # Object names from JSON
        'class_labels': List[str],              # Class labels from aggregation.json
        'num_objects': int,                     # Total number of objects (e.g., 7)
        'num_instances': int,                   # Number of unique classes (e.g., 2)
        'gt_pmasks': List[Tensor[N]],           # Binary masks for each object
        'text_input': str,                      # Question text
        'answer': str,                          # Answer text
        'data_type': str                        # Question type
    }

Output Format (Batch):
    {
        'coords': Tensor[B*N, 4],               # Concatenated coordinates with batch index
        'feats': Tensor[B*N, 3],                # Concatenated RGB features
        'object_ids': List[List[int]],          # List of object ID lists per sample
        'class_labels': List[List[str]],        # List of class label lists per sample
        'num_objects': List[int],               # Object counts per sample
        'num_instances': List[int],             # Unique class counts per sample
        'gt_pmasks': List[List[Tensor]],        # List of mask lists per sample
        ...
    }

Key Features:
    - No downsampling: Full point clouds preserved
    - Masks generated from aggregation.json segments field
    - Multi-object samples kept together (not split)
    - num_objects: total object count, num_instances: unique class count
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
from typing import Dict, List, Tuple, Optional
import warnings


class MultiScanDataset(Dataset):
    """
    Multi-Scan Dataset for Multi-Object Instance Reasoning Segmentation
    Returns multiple object masks and classes for each sample
    
    Args:
        reasoning_dir: Directory path containing Multi-Scan JSON files
        scans_dir: Directory path containing 3D point cloud scans
        load_point_cloud: Whether to load point cloud data
        normalize: Whether to normalize point cloud coordinates
        use_xyz: Whether to concatenate xyz to features
        mode: Voxelization mode (default: 4)
        training: Whether in training mode (for data augmentation)
    """
    
    def __init__(
        self,
        reasoning_dir: str,
        scans_dir: str,
        load_point_cloud: bool = True,
        normalize: bool = True,
        use_xyz: bool = True,
        mode: int = 4,
        training: bool = True,
        aug: bool = True
    ):
        self.reasoning_dir = reasoning_dir
        self.scans_dir = scans_dir
        self.load_point_cloud = load_point_cloud
        self.normalize = normalize
        self.use_xyz = use_xyz
        self.mode = mode
        self.training = training
        self.aug = aug
        
        # Load all data at once
        self.data = self._load_data()
        
        print(f"Loaded {len(self.data)} samples from Multi-Scan dataset (multi-object format)")
    
    def _load_data(self) -> List[Dict]:
        """Load all JSON data files"""
        data = []
        
        # Load all Multi-Scan JSON files
        json_files = [
            "Multi-Scan-1-v2.json",
            "Multi-Scan-2-v2.json",
            "Multi-Scan-3-v2.json"
        ]
        
        # Load each JSON file
        for json_file in json_files:
            json_path = os.path.join(self.reasoning_dir, json_file)
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    data.extend(file_data)
                print(f"Loaded {len(file_data)} samples from {json_file}")
            else:
                warnings.warn(f"File not found: {json_path}")
        
        return data
    
    def _parse_object_ids(self, obj_id_str) -> List[int]:
        """Parse object_id string, handling comma-separated multiple IDs"""
        if isinstance(obj_id_str, int):
            return [obj_id_str]
        elif isinstance(obj_id_str, str):
            return [int(x.strip()) for x in obj_id_str.split(',')]
        else:
            return []
    
    def _parse_all_objects(self, item: Dict) -> Tuple[List[int], List[str]]:
        """
        Parse all objects in a sample, maintaining their association
        Returns: (object_ids, object_names)
        object_names here are descriptive names from JSON, not class labels
        """
        object_ids = []
        object_names = []
        
        if 'object_id_1' in item:
            obj_ids = self._parse_object_ids(item['object_id_1'])
            object_ids.extend(obj_ids)
            object_names.extend([item['object_id_1_name']] * len(obj_ids))
        
        if 'object_id_2' in item:
            obj_ids = self._parse_object_ids(item['object_id_2'])
            object_ids.extend(obj_ids)
            object_names.extend([item['object_id_2_name']] * len(obj_ids))
        
        if 'object_id_3' in item:
            obj_ids = self._parse_object_ids(item['object_id_3'])
            object_ids.extend(obj_ids)
            object_names.extend([item['object_id_3_name']] * len(obj_ids))
        
        return object_ids, object_names
    
    def _load_point_cloud(self, scene_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load point cloud data for a scene
        Returns: (xyz, rgb, semantic_label)
        """
        ply_path = os.path.join(self.scans_dir, scene_id, f"{scene_id}_vh_clean_2.labels.ply")
        
        if not os.path.exists(ply_path):
            warnings.warn(f"Point cloud file not found: {ply_path}")
            return None, None, None
        
        try:
            # Read PLY file
            plydata = PlyData.read(ply_path)
            vertices = plydata['vertex']
            
            # Extract coordinates (x, y, z)
            xyz = np.vstack([
                vertices['x'],
                vertices['y'],
                vertices['z']
            ]).T.astype(np.float32)
            
            # Extract color information
            rgb = None
            if 'red' in vertices.data.dtype.names:
                rgb = np.vstack([
                    vertices['red'],
                    vertices['green'],
                    vertices['blue']
                ]).T.astype(np.float32) / 255.0  # Normalize to [0,1]
            else:
                rgb = np.ones((xyz.shape[0], 3), dtype=np.float32)
            
            # Extract semantic label (use instance label as proxy)
            semantic_label = None
            if 'label' in vertices.data.dtype.names:
                semantic_label = vertices['label'].astype(np.int32)
            else:
                semantic_label = np.zeros(xyz.shape[0], dtype=np.int32)
            
            return xyz, rgb, semantic_label
            
        except Exception as e:
            warnings.warn(f"Error loading point cloud {ply_path}: {e}")
            return None, None, None
    
    def _load_segmentation(self, scene_id: str) -> Optional[np.ndarray]:
        """Load segmentation indices for each point"""
        seg_path = os.path.join(self.scans_dir, scene_id, f"{scene_id}_vh_clean_2.0.010000.segs.json")
        
        if not os.path.exists(seg_path):
            warnings.warn(f"Segmentation file not found: {seg_path}")
            return None
        
        try:
            with open(seg_path, 'r') as f:
                seg_data = json.load(f)
                seg_indices = np.array(seg_data['segIndices'], dtype=np.int32)
                return seg_indices
        except Exception as e:
            warnings.warn(f"Error loading segmentation {seg_path}: {e}")
            return None
    
    def _load_aggregation(self, scene_id: str) -> Optional[Dict]:
        """Load aggregation information for a scene"""
        agg_path = os.path.join(self.scans_dir, scene_id, f"{scene_id}.aggregation.json")
        
        if not os.path.exists(agg_path):
            return None
        
        try:
            with open(agg_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"Error loading aggregation {agg_path}: {e}")
            return None
    
    def _generate_multi_object_masks(self, object_ids: List[int], aggregation: Dict, 
                                       seg_indices: np.ndarray, num_points: int) -> Tuple[List[torch.Tensor], List[str], torch.Tensor]:
        """
        Generate point-wise masks for multiple objects and extract class labels
        
        Args:
            object_ids: List of object IDs to generate masks for
            aggregation: Aggregation data containing object-to-segment mapping
            seg_indices: Segment index for each point [N]
            num_points: Total number of points
            
        Returns:
            gt_pmasks: List of point-wise binary masks, one per object
            class_labels: List of class labels (from aggregation.json), one per object
            superpoint: Superpoint indices [N] (using segment indices as proxy)
        """
        if aggregation is None or seg_indices is None:
            dummy_masks = [torch.zeros(num_points, dtype=torch.bool) for _ in object_ids]
            dummy_labels = ['unknown'] * len(object_ids)
            return dummy_masks, dummy_labels, torch.zeros(num_points, dtype=torch.long)
        
        # Build object_id to segments and label mapping from aggregation.json
        object_to_segments = {}
        object_to_label = {}
        for seg_group in aggregation['segGroups']:
            obj_id = seg_group['objectId']
            segments = seg_group['segments']  # This is the segments field from aggregation.json
            label = seg_group.get('label', 'unknown')  # Class label from aggregation.json
            object_to_segments[obj_id] = segments
            object_to_label[obj_id] = label
        
        # Generate mask and get class label for each object
        gt_pmasks = []
        class_labels = []
        for obj_id in object_ids:
            mask = np.zeros(num_points, dtype=np.bool_)
            
            if obj_id in object_to_segments:
                # Get all segment IDs for this object from aggregation.json
                segments = object_to_segments[obj_id]
                
                # Mark points belonging to these segments
                for seg_id in segments:
                    mask[seg_indices == seg_id] = True
                
                # Get class label from aggregation.json
                class_label = object_to_label[obj_id]
            else:
                class_label = 'unknown'
            
            gt_pmasks.append(torch.from_numpy(mask).bool())
            class_labels.append(class_label)
        
        # Use segment indices as superpoint
        superpoint = torch.from_numpy(seg_indices).long()
        
        return gt_pmasks, class_labels, superpoint
    
    def transform_train(self, xyz, rgb):
        """Data augmentation for training"""
        # Normalize coordinates
        xyz_max = np.max(xyz, axis=0, keepdims=True)
        xyz = xyz - xyz_max
        
        if self.aug:
            # Apply data augmentation
            xyz = self._data_aug(xyz, jitter=True, flip=True, rot=True)
        
        # Add noise to RGB
        rgb = rgb + np.random.randn(3) * 0.1
        rgb = np.clip(rgb, 0, 1)
        
        # Scale coordinates
        xyz_middle = xyz.copy()
        xyz = xyz_middle * 50
        xyz = xyz - xyz.min(0)
        
        return xyz, xyz_middle, rgb
    
    def transform_test(self, xyz, rgb):
        """Transform for testing"""
        xyz_max = np.max(xyz, axis=0, keepdims=True)
        xyz = xyz - xyz_max
        xyz_middle = xyz.copy()
        xyz = xyz_middle * 50
        xyz = xyz - xyz.min(0)
        return xyz, xyz_middle, rgb
    
    def _data_aug(self, xyz, jitter=False, flip=False, rot=False):
        """Data augmentation"""
        import math
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [[math.cos(theta), math.sin(theta), 0], 
                 [-math.sin(theta), math.cos(theta), 0], 
                 [0, 0, 1]])
        return np.matmul(xyz, m)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Return a data sample with MULTIPLE objects and their masks
        
        Returns:
            - coord: Integer coordinates [N, 3]
            - coord_float: Float coordinates [N, 3]
            - feat: RGB features [N, 3]
            - superpoint: Superpoint indices [N]
            - object_ids: List of object IDs (M objects)
            - object_names: List of object names (M objects)
            - gt_pmasks: List of point-wise masks [M x N]
            - text_input: Question string
            - answers: List of answer strings
        """
        item = self.data[idx]
        
        scene_id = item['scene_id']
        question_desc = item['question']
        data_type = item['type']
        
        # Parse all objects in this sample
        object_ids, object_names = self._parse_all_objects(item)
        
        # Load point cloud
        xyz, rgb, semantic_label = self._load_point_cloud(scene_id)
        
        if xyz is None:
            # Return dummy data if loading fails
            dummy_size = 100
            dummy_masks = [torch.zeros(dummy_size, dtype=torch.bool) for _ in object_ids]
            dummy_labels = ['unknown'] * len(object_ids)
            return {
                'ann_ids': idx,
                'scan_ids': scene_id,
                'coord': torch.zeros((dummy_size, 3), dtype=torch.long),
                'coord_float': torch.zeros((dummy_size, 3), dtype=torch.float32),
                'feat': torch.zeros((dummy_size, 3), dtype=torch.float32),
                'superpoint': torch.zeros(dummy_size, dtype=torch.long),
                'object_ids': object_ids,
                'object_names': object_names,
                'class_labels': dummy_labels,
                'num_objects': len(object_ids),
                'num_instances': len(set(dummy_labels)),
                'gt_pmasks': dummy_masks,
                'gt_spmask': None,
                'answer': item['answer'],
                'text_input': question_desc,
                'data_type': data_type,
            }
        
        # No downsampling - use full point cloud
        # Store original indices for potential future use
        indices = np.arange(xyz.shape[0])
        
        # Apply transforms
        if self.training:
            xyz, xyz_middle, rgb = self.transform_train(xyz, rgb)
        else:
            xyz, xyz_middle, rgb = self.transform_test(xyz, rgb)
        
        # Convert to tensors
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()
        
        # Load segmentation and generate masks for ALL objects
        seg_indices = self._load_segmentation(scene_id)
        aggregation = self._load_aggregation(scene_id)
        
        if seg_indices is not None and aggregation is not None:
            # Ensure seg_indices matches point cloud size
            if seg_indices.shape[0] != xyz.shape[0]:
                warnings.warn(f"Segmentation size {seg_indices.shape[0]} != point cloud size {xyz.shape[0]}")
                # Truncate or pad to match
                if seg_indices.shape[0] > xyz.shape[0]:
                    seg_indices = seg_indices[:xyz.shape[0]]
                else:
                    padding = np.zeros(xyz.shape[0] - seg_indices.shape[0], dtype=np.int32)
                    seg_indices = np.concatenate([seg_indices, padding])
            
            # Generate masks and class labels for all objects based on segments field
            gt_pmasks, class_labels, superpoint = self._generate_multi_object_masks(
                object_ids, aggregation, seg_indices, xyz.shape[0]
            )
        else:
            gt_pmasks = [torch.zeros(xyz.shape[0], dtype=torch.bool) for _ in object_ids]
            class_labels = ['unknown'] * len(object_ids)
            superpoint = torch.zeros(xyz.shape[0], dtype=torch.long)
        
        # Use original question directly
        text_input = question_desc
        answer = item['answer']
        
        # Count unique class labels for instance count
        unique_classes = list(set(class_labels))
        num_instances = len(unique_classes)
        
        return {
            'ann_ids': idx,
            'scan_ids': scene_id,
            'coord': coord,
            'coord_float': coord_float,
            'feat': feat,
            'superpoint': superpoint,
            'object_ids': object_ids,  # List of object IDs (e.g., [0, 1, 2, 3, 4, 26, 16])
            'object_names': object_names,  # List of descriptive names from JSON
            'class_labels': class_labels,  # List of class labels from aggregation.json (e.g., ['cabinet', 'cabinet', ...])
            'num_objects': len(object_ids),  # Total number of objects (e.g., 7)
            'num_instances': num_instances,  # Number of unique classes (e.g., 3)
            'gt_pmasks': gt_pmasks,  # List of masks based on segments from aggregation.json
            'gt_spmask': None,
            'sp_ref_mask': None,
            'answer': answer,
            'text_input': text_input,
            'data_type': data_type,
        }


def collate_multi_scan(batch):
    """
    Collate function for multi-object format
    Each sample contains multiple objects with their masks
    """
    ann_ids, scan_ids, coords, coords_float, feats, superpoints = [], [], [], [], [], []
    all_object_ids, all_object_names, all_class_labels, all_gt_pmasks = [], [], [], []
    all_num_objects, all_num_instances = [], []
    answers_list, text_input_list, data_type_list = [], [], []
    batch_offsets = [0]
    superpoint_bias = 0
    
    for i, data in enumerate(batch):
        # Extract data
        ann_id = data['ann_ids']
        scan_id = data['scan_ids']
        coord = data['coord']
        coord_float = data['coord_float']
        feat = data['feat']
        src_superpoint = data['superpoint']
        object_ids_sample = data['object_ids']  # List of IDs for this sample
        object_names_sample = data['object_names']  # List of names for this sample
        class_labels_sample = data['class_labels']  # List of class labels for this sample
        gt_pmasks_sample = data['gt_pmasks']  # List of masks for this sample
        num_objects = data['num_objects']
        num_instances = data['num_instances']
        answer = data['answer']
        text_input = data['text_input']
        data_type = data['data_type']
        
        # Add batch index to coordinates
        superpoint = src_superpoint + superpoint_bias
        superpoint_bias = superpoint.max().item() + 1
        batch_offsets.append(superpoint_bias)
        
        # Collect data
        ann_ids.append(ann_id)
        scan_ids.append(scan_id)
        coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))
        coords_float.append(coord_float)
        feats.append(feat)
        superpoints.append(superpoint)
        all_object_ids.append(object_ids_sample)  # Keep as list
        all_object_names.append(object_names_sample)  # Keep as list
        all_class_labels.append(class_labels_sample)  # Keep as list
        all_gt_pmasks.append(gt_pmasks_sample)  # Keep as list of masks
        all_num_objects.append(num_objects)
        all_num_instances.append(num_instances)
        answers_list.append(answer)
        text_input_list.append(text_input)
        data_type_list.append(data_type)
    
    # Concatenate batch
    batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)
    coords = torch.cat(coords, 0)  # [B*N, 4] (batch_idx, x, y, z)
    coords_float = torch.cat(coords_float, 0)  # [B*N, 3]
    feats = torch.cat(feats, 0)  # [B*N, 3]
    superpoints = torch.cat(superpoints, 0).long()  # [B*N]
    
    return {
        'ann_ids': ann_ids,
        'scan_ids': scan_ids,
        'coords': coords,
        'coords_float': coords_float,
        'feats': feats,
        'superpoints': superpoints,
        'batch_offsets': batch_offsets,
        'object_ids': all_object_ids,  # List of lists: [[obj1, obj2], [obj3], ...]
        'object_names': all_object_names,  # List of lists: [['chair', 'table'], ['sofa'], ...]
        'class_labels': all_class_labels,  # List of lists: [['cabinet', 'cabinet', 'clothes dryer'], ...]
        'num_objects': all_num_objects,  # List of ints: [7, 4, ...]
        'num_instances': all_num_instances,  # List of ints: [3, 2, ...]
        'gt_pmasks': all_gt_pmasks,  # List of lists of masks: [[[N], [N]], [[N]], ...]
        'answers': answers_list,  # List of answer strings
        'text_input': text_input_list,
        'data_types': data_type_list,
    }


if __name__ == "__main__":
    # Test code
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MultiScanDataset')
    parser.add_argument('--reasoning_dir', type=str, required=True,
                        help='Directory containing Multi-Scan JSON files')
    parser.add_argument('--scans_dir', type=str, required=True,
                        help='Directory containing scans')
    args = parser.parse_args()
    
    print("Testing MultiScanDataset...")
    
    # Create dataset - no downsampling, full point cloud
    dataset = MultiScanDataset(
        reasoning_dir=args.reasoning_dir,
        scans_dir=args.scans_dir,
        load_point_cloud=True,
        training=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print(f"\nSample 0 (Multi-object format):")
    print(f"  Scene ID: {sample['scan_ids']}")
    print(f"  Number of objects: {sample['num_objects']}")
    print(f"  Number of instances (unique classes): {sample['num_instances']}")
    print(f"  Object IDs: {sample['object_ids']}")
    print(f"  Object Names (from JSON): {sample['object_names']}")
    print(f"  Class Labels (from aggregation.json): {sample['class_labels']}")
    print(f"  Coord shape: {sample['coord'].shape}")
    print(f"  Coord_float shape: {sample['coord_float'].shape}")
    print(f"  Feat shape: {sample['feat'].shape}")
    print(f"  Superpoint shape: {sample['superpoint'].shape}")
    print(f"  Number of GT masks: {len(sample['gt_pmasks'])}")
    for i, mask in enumerate(sample['gt_pmasks']):
        print(f"    Mask {i} - Object ID {sample['object_ids'][i]} - Class '{sample['class_labels'][i]}': {mask.sum().item()}/{len(mask)} points")
    print(f"  Question: {sample['text_input'][:100]}...")
    print(f"  Answer: {sample['answer'][:100]}...")
    
    # Test DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_multi_scan
    )
    
    print("\nTesting DataLoader...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Batch size: {len(batch['scan_ids'])}")
        print(f"  Scene IDs: {batch['scan_ids']}")
        print(f"  Coords shape: {batch['coords'].shape}")
        print(f"  Feats shape: {batch['feats'].shape}")
        print(f"  Multi-object info:")
        for i, (obj_ids, obj_names, class_labels, masks, num_objs, num_insts) in enumerate(zip(
            batch['object_ids'], 
            batch['object_names'],
            batch['class_labels'],
            batch['gt_pmasks'],
            batch['num_objects'],
            batch['num_instances']
        )):
            print(f"    Sample {i}: {num_objs} objects, {num_insts} instances (unique classes)")
            print(f"      Object IDs: {obj_ids}")
            print(f"      Class Labels: {class_labels}")
            unique_classes = list(set(class_labels))
            print(f"      Unique Classes: {unique_classes}")
        
        if batch_idx >= 1:  # Show first 2 batches
            break
    
    print("\nTest completed!")
