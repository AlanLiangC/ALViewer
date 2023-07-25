from pathlib import Path
import numpy as np
import torch
from torch import Tensor


KITTI = '/home/alan/AlanLiang/Projects/3D_Perception/mmdetection3d/data/kitti'


def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)


def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / img_idx_str
    else:
        file_path = Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_calib_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check, use_prefix_id)

def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def get_kitti_calib_info(idx, 
                         path, 
                         training=True, 
                         relative_path=False,
                         extend_matrix=True):
    
    calib_info = {}
    calib_path = get_calib_path(
                idx, path, training, relative_path)
    
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                    ]).reshape([3, 4])
    P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                    ]).reshape([3, 4])
    P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                    ]).reshape([3, 4])
    P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                    ]).reshape([3, 4])
        
    if extend_matrix:
        P0 = _extend_matrix(P0)
        P1 = _extend_matrix(P1)
        P2 = _extend_matrix(P2)
        P3 = _extend_matrix(P3)

    R0_rect = np.array([
                float(info) for info in lines[4].split(' ')[1:10]
            ]).reshape([3, 3])
    if extend_matrix:
        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.
        rect_4x4[:3, :3] = R0_rect
    else:
        rect_4x4 = R0_rect

    Tr_velo_to_cam = np.array([
                float(info) for info in lines[5].split(' ')[1:13]
            ]).reshape([3, 4])
    Tr_imu_to_velo = np.array([
        float(info) for info in lines[6].split(' ')[1:13]
    ]).reshape([3, 4])
    if extend_matrix:
        Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
        Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
    calib_info['P0'] = P0
    calib_info['P1'] = P1
    calib_info['P2'] = P2
    calib_info['P3'] = P3
    calib_info['R0_rect'] = rect_4x4
    calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
    calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo

    return calib_info

def kitti_cam2lidar(points, r_rect, velo2cam):
    """Convert points in camera coordinate to lidar coordinate.

    Note:
        This function is for KITTI only.

    Args:
        points (np.ndarray, shape=[N, 3]): Points in camera coordinate.
        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.

    Returns:
        np.ndarray, shape=[N, 3]: Points in lidar coordinate.
    """
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]

def limit_period(val: np.ndarray,
                 offset: float = 0.5,
                 period: float = np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (np.ndarray or Tensor): The value to be converted.
        offset (float): Offset to set the value range. Defaults to 0.5.
        period (float): Period of the value. Defaults to np.pi.

    Returns:
        np.ndarray or Tensor: Value in the range of
        [-offset * period, (1-offset) * period].
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def kitti_img2cam(
        points: Tensor,
        cam2img: Tensor):
    """Project points in image coordinates to camera coordinates.

    Args:
        points (Tensor or np.ndarray): 2.5D points in 2D images with shape
            [N, 3], 3 corresponds with x, y in the image and depth.
        cam2img (Tensor or np.ndarray): Camera intrinsic matrix. The shape can
            be [3, 3], [3, 4] or [4, 4].

    Returns:
        Tensor or np.ndarray: Points in 3D space with shape [N, 3], 3
        corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D

def kitti_img2lidar(calib_info: dict, points: np.ndarray):
    """Project points in image coordinates to camera coordinates.

    Args:
        points (Tensor or np.ndarray): 2.5D points in 2D images with shape
            [N, 3], 3 corresponds with x, y in the image and depth.
        calib_info (Dict): Camera information.

    Returns:
        Tensor or np.ndarray: Points in 3D space with shape [N, 3], 3
        corresponds with x, y, z in 3D lidar space.
    """

    cam2img = calib_info['P2'] # only CAM2
    r_rect = calib_info['R0_rect']
    velo2cam = calib_info['Tr_velo_to_cam']
    cam_points = kitti_img2cam(points=points, cam2img=cam2img)
    lidar_points = kitti_cam2lidar(points=cam_points, r_rect=r_rect, velo2cam=velo2cam) 

    return lidar_points



if __name__ == "__main__":

    calib_info = get_kitti_calib_info(idx=0, path=KITTI)
    print(calib_info.keys())