'''
Fog
'''
def fog_sim(pointcloud,severity):

    from tools.lidar_corruption.fog_sim import simulate_fog, ParameterSet
    c = [0.005, 0.01, 0.02, 0.03, 0.06][severity-1] # form original paper
    parameter_set = ParameterSet(alpha=c, gamma=0.000001)
    points, _, _ = simulate_fog(parameter_set, pointcloud, noise=10)
    return points