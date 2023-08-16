'''
Fog
'''
def fog_sim(pointcloud,severity):

    from tools.lidar_corruption.fog_sim import simulate_fog, ParameterSet
    c = [0.005, 0.01, 0.02, 0.03, 0.06][severity-1] # form original paper
    parameter_set = ParameterSet(alpha=c*10, gamma=0.000001)
    points, _, _ = simulate_fog(parameter_set, pointcloud, noise=10)
    return points

'''
Rain
'''

def rain_sim(pointcloud,severity):
    from tools.lidar_corruption.lisa import LISA
    rain_sim = LISA(show_progressbar=True)
    c = [0.20, 0.73, 1.5625, 3.125, 7.29, 10.42][severity-1]
    points = rain_sim.augment(pointcloud, c*5)
    return points