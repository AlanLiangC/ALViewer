         #   R,   G,   B, alpha
COLORS = [(  0, 255,   0, 255),  # cars in green
          (255,   0,   0, 255),  # pedestrian in red
          (255, 255,   0, 255),  # cyclists in yellow
          (255, 127,  80, 255),  # Coral
          (233, 150,  70, 255),  # Darksalmon
          (220,  20,  60, 255),  # Crimson
          (255,  61,  99, 255),  # Red
          (0,     0, 230, 255),  # Blue
          (47,   79,  79, 255),  # Darkslategrey
         (112,  128, 144, 255),  # Slategrey
        ]

NUSCENES_SEMANTIC_INFO = {
        'classes':
        ('noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'),
        'ignore_index':
        0,
        'label_mapping':
        dict([(1, 0), (5, 0), (7, 0), (8, 0), (10, 0), (11, 0), (13, 0),
              (19, 0), (20, 0), (0, 0), (29, 0), (31, 0), (9, 1), (14, 2),
              (15, 3), (16, 3), (17, 4), (18, 5), (21, 6), (2, 7), (3, 7),
              (4, 7), (6, 7), (12, 8), (22, 9), (23, 10), (24, 11), (25, 12),
              (26, 13), (27, 14), (28, 15), (30, 16)]),
        'palette': [
            [0, 0, 0],  # noise
            [255, 120, 50],  # barrier              orange
            [255, 192, 203],  # bicycle              pink
            [255, 255, 0],  # bus                  yellow
            [0, 150, 245],  # car                  blue
            [0, 255, 255],  # construction_vehicle cyan
            [255, 127, 0],  # motorcycle           dark orange
            [255, 0, 0],  # pedestrian           red
            [255, 240, 150],  # traffic_cone         light yellow
            [135, 60, 0],  # trailer              brown
            [160, 32, 240],  # truck                purple
            [255, 0, 255],  # driveable_surface    dark pink
            [139, 137, 137],  # other_flat           dark red
            [75, 0, 75],  # sidewalk             dard purple
            [150, 240, 80],  # terrain              light green
            [230, 230, 250],  # manmade              white
            [0, 175, 0],  # vegetation           green
        ]
    }

color_dict = {0: 'x',
              1: 'y',
              2: 'z',
              3: 'intensity',
              4: 'distance',
              5: 'angle',
              6: 'channel'}

grid_dimensions = 20
success = False # ?
min_value = 0
max_value = 63
num_features = 5
color_feature = 2
point_size = 3
extension = 'bin'
intensity_multiplier = 1
color_name = color_dict[color_feature]


FOG = 'DENSE/SeeingThroughFog/lidar_hdl64_strongest_fog_extraction'
AUDI = 'A2D2/camera_lidar_semantic_bboxes'
LYFT = 'LyftLevel5/Perception/train_lidar'
ARGO = 'Argoverse'
PANDA = 'PandaSet'
DENSE = 'DENSE/SeeingThroughFog/lidar_hdl64_strongest'
KITTI = '/home/alan/AlanLiang/Projects/3D_Perception/mmdetection3d/data/kitti'
WAYMO = 'WaymoOpenDataset/WOD/train/velodyne'
HONDA = 'Honda_3D/scenarios'
APOLLO = 'Apollo3D'
NUSCENES = '/home/alan/AlanLiang/Projects/3D_Perception/mmdetection3d/data/nuscenes'