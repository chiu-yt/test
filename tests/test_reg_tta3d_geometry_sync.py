import ast
import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = REPO_ROOT / 'pcdet' / 'tta_methods' / 'reg_tta3d_geometry.py'


def _tree():
    assert GEOMETRY_PATH.is_file(), 'Reg-TTA3D geometry module is missing: %s' % GEOMETRY_PATH
    return ast.parse(GEOMETRY_PATH.read_text(encoding='utf-8'), filename=str(GEOMETRY_PATH))


def _source(node):
    source = ast.get_source_segment(GEOMETRY_PATH.read_text(encoding='utf-8'), node)
    assert source is not None
    return source


def _student_view_function():
    matches = [
        node for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == 'build_reg_tta3d_student_view'
    ]
    assert len(matches) == 1, 'build_reg_tta3d_student_view must be defined exactly once'
    return matches[0]


def _runtime_geometry():
    getattr(importlib.import_module('pytest'), 'importorskip')('torch')
    return importlib.import_module('pcdet.tta_methods.reg_tta3d_geometry')


def test_student_view_static_contract_synchronizes_multimodal_geometry_and_revoxels():
    # Given the planned student-view construction source.
    function = _student_view_function()

    # When it is inspected through stdlib AST only.
    source = _source(function)

    # Then one geometry callback runs before voxelization and stale voxel tensors are removed first.
    assert source.count('augment_student_geometry(') == 1
    assert source.count('voxelize(') == 1
    assert source.index("pop(voxel_field, None)") < source.index('augment_student_geometry(')
    assert source.index('augment_student_geometry(') < source.index('voxelize(')
    assert all(field in source for field in ('voxels', 'voxel_coords', 'voxel_num_points'))

    # Then cached teacher voxel tensors cannot be assigned directly into the student view.
    copied_voxels = [
        _source(node) for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any('voxel' in _source(target) for target in node.targets)
        and 'teacher_batch' in _source(node.value)
    ]
    assert copied_voxels == []


def test_production_builder_separates_labels_and_masks_augmented_points():
    # Given the production builder source without importing its runtime dependencies.
    source = GEOMETRY_PATH.read_text(encoding='utf-8')

    # When pseudo-target and point preprocessing are inspected.
    label_split = source.index("pseudo_targets[batch_index, valid_targets, -1]")
    geometry_split = source.index("pseudo_targets[batch_index, valid_targets, :-1]")
    range_mask = source.index('mask_points_by_range')
    voxelization = source.index('self.voxel_processor(data_dict=sample)')

    # Then labels are handled outside augmentation and point masking precedes voxelization.
    assert label_split < source.index('target_labels.to')
    assert geometry_split < source.index('self.augmentor.forward')
    assert range_mask < voxelization


def test_student_view_runtime_applies_one_geometry_transform_and_regenerates_voxels():
    # Given a teacher batch with geometric fields and stale voxel tensors.
    geometry = _runtime_geometry()
    torch = importlib.import_module('torch')
    teacher_batch = {
        'points': torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
        'gt_boxes': torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]]]),
        'tta_proposal_boxes': torch.tensor([[[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.1]]]),
        'lidar_aug_matrix': torch.eye(4).unsqueeze(0),
        'img_aug_matrix': torch.eye(4).unsqueeze(0),
        'lidar2camera': torch.eye(4).unsqueeze(0),
        'lidar2image': torch.eye(4).unsqueeze(0),
        'camera2ego': torch.eye(4).unsqueeze(0),
        'camera_intrinsics': torch.eye(4).unsqueeze(0),
        'camera2lidar': torch.eye(4).unsqueeze(0),
        'voxels': torch.full((1, 1), -1.0),
        'voxel_coords': torch.full((1, 4), -1, dtype=torch.int32),
        'voxel_num_points': torch.full((1,), -1, dtype=torch.int32),
    }

    def augment_student_geometry(batch):
        student = dict(batch)
        student['points'] = batch['points'] + 10.0
        student['gt_boxes'] = batch['gt_boxes'] + 10.0
        student['tta_proposal_boxes'] = batch['tta_proposal_boxes'] + 10.0
        student['lidar_aug_matrix'] = batch['lidar_aug_matrix'] * 2.0
        student['img_aug_matrix'] = batch['img_aug_matrix'] * 3.0
        for field in ('lidar2camera', 'lidar2image', 'camera2ego', 'camera_intrinsics', 'camera2lidar'):
            student[field] = batch[field] * 4.0
        return student

    def voxelize(points):
        return {
            'voxels': points[:, 1:].unsqueeze(1),
            'voxel_coords': torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
            'voxel_num_points': torch.tensor([points.shape[0]], dtype=torch.int32),
        }

    # When one student geometric augmentation is built from the teacher view.
    student_batch = geometry.build_reg_tta3d_student_view(
        teacher_batch, augment_student_geometry, voxelize,
    )

    # Then every geometric sidecar agrees with the augmented points/boxes and voxels are fresh.
    torch.testing.assert_close(student_batch['points'], teacher_batch['points'] + 10.0)
    torch.testing.assert_close(student_batch['gt_boxes'], teacher_batch['gt_boxes'] + 10.0)
    torch.testing.assert_close(student_batch['tta_proposal_boxes'], teacher_batch['tta_proposal_boxes'] + 10.0)
    torch.testing.assert_close(student_batch['lidar_aug_matrix'], teacher_batch['lidar_aug_matrix'] * 2.0)
    torch.testing.assert_close(student_batch['img_aug_matrix'], teacher_batch['img_aug_matrix'] * 3.0)
    for field in ('lidar2camera', 'lidar2image', 'camera2ego', 'camera_intrinsics', 'camera2lidar'):
        torch.testing.assert_close(student_batch[field], teacher_batch[field] * 4.0)
    assert student_batch['voxels'] is not teacher_batch['voxels']
    torch.testing.assert_close(student_batch['voxels'], student_batch['points'][:, 1:].unsqueeze(1))
    assert student_batch['voxel_coords'] is not teacher_batch['voxel_coords']
    assert student_batch['voxel_num_points'] is not teacher_batch['voxel_num_points']
