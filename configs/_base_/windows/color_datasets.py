         #   R,   G,   B, alpha
COLORS = [(  0, 255,   0, 255),  # cars in green
          (255,   0,   0, 255),  # pedestrian in red
          (255, 255,   0, 255),  # cyclists in yellow
          (  0, 255,   0, 255),  
          (255,   0,   0, 255), 
          (255, 255,   0, 255), 

color_dict = {0: 'x',
              1: 'y',
              2: 'z',
              3: 'intensity',
              4: 'distance',
              5: 'angle',
              6: 'channel'}

grid_dimensions = 20
dataset = None
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