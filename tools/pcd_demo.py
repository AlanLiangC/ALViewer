# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--pcd', help='Point cloud file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        default=True,
        # action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        default=False,
        help='whether to save online visualization results')
    args = parser.parse_args()
    return args


def main(args):
    # TODO: Support inference of point cloud numpy file.
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # test a single point cloud sample
    result, data = inference_detector(model, args.pcd)
    points = data['inputs']['points']
    data_input = dict(points=points)

    # show the results
    visualizer.add_datasample(
        'result',
        data_input,
        data_sample=result,
        draw_gt=False,
        show=args.show,
        wait_time=-1,
        out_file=args.out_dir,
        pred_score_thr=args.score_thr,
        vis_task='lidar_det')


if __name__ == '__main__':
    args = parse_args()


    args.config = '/home/alan/AlanLiang/Projects/3D_Perception/ALViewer/detection_tasks/configs/centerpoint_pillar02_second_secfpn_head-dcn_8xb4-cyclic-20e_nus-3d.py'
    args.checkpoint = '/home/alan/AlanLiang/Projects/3D_Perception/ALViewer/detection_tasks/pretrained_models/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_20220811_045458-808e69ad.pth'
    # args.pcd = '/home/alan/AlanLiang/Projects/3D_Perception/ALViewer/000008.bin'
    args.pcd = '/home/alan/AlanLiang/Projects/3D_Perception/mmdetection3d/data/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915245048008.pcd.bin'

    main(args)
