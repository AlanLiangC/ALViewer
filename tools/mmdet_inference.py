import torch
from mmdet3d.apis import inference_detector, init_model
from windows.common import tensor2ndarray


def mmdet_inference(cfgs):
    # TODO: Support inference of point cloud numpy file.
    # build the model from a config file and a checkpoint file
    model = init_model(cfgs.config, cfgs.checkpoint, device=cfgs.device)

    result, data = inference_detector(model, cfgs.pcd)
    points = data['inputs']['points']
    data_input = dict(points=points)

    pred_instances_3d = result.pred_instances_3d
    pred_instances_3d = pred_instances_3d[
                    pred_instances_3d.scores_3d > cfgs['score_thr']].to('cpu')

    try:
        bboxes_3d = pred_instances_3d.bboxes_3d.tensor.view(-1,7)  # KITTI
    except:
        bboxes_3d = pred_instances_3d.bboxes_3d.tensor.view(-1,9)  # Nuscenes
    
    labels_3d = pred_instances_3d.labels_3d.view(-1,1) + 1

    result = torch.cat([bboxes_3d, labels_3d], dim = -1)
    result = tensor2ndarray(result)
    
    return result

if __name__ == '__main__':
    from mmengine.config import Config

    det_task_config = Config.fromfile('/home/alan/AlanLiang/Projects/3D_Perception/ALViewer/configs/detection_window/detection_window.py')

    det_task_config['config'] = '/home/alan/AlanLiang/Projects/3D_Perception/ALViewer/detection_tasks/configs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
    
    det_task_config['checkpoint'] = '/home/alan/AlanLiang/Projects/3D_Perception/ALViewer/detection_tasks/pretrained_models/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
    
    # det_task_config['pcd'] = '/home/alan/AlanLiang/Projects/3D_Perception/mmdetection3d/data/kitti/training/velodyne/000010.bin'
    det_task_config['pcd'] = '/home/alan/AlanLiang/Projects/3D_Perception/ALViewer/000008.bin'
    result = mmdet_inference(det_task_config)

    print(result)


