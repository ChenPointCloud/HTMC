import open3d as o3d
import torch


def compare(c1, c2):
    if torch.is_tensor(c1):
        c1 = c1.detach().cpu().numpy()
        c2 = c2.detach().cpu().numpy()
    if c1.shape[0]==1:
        c1=c1[0]
        c2=c2[0]
    if c1.shape[0]==3:
        c1=c1.T
        c2=c2.T
    cloud1 = o3d.geometry.PointCloud()
    cloud1.points = o3d.utility.Vector3dVector(c1)
    cloud2 = o3d.geometry.PointCloud()
    cloud2.points = o3d.utility.Vector3dVector(c2)

    cloud1.paint_uniform_color([1, 0, 0])
    cloud2.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([cloud1, cloud2],  # 待显示的点云列表
                                      window_name="点云显示",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)

def compare3(c1, c2, c3):
    if c1.shape[0] == 1:
        c1 = c1[0]
        c2 = c2[0]
        c3 = c3[0]
    if c1.shape[0] == 3:
        c1 = c1.T
        c2 = c2.T
        c3 = c3.T
    cloud1 = o3d.geometry.PointCloud()
    cloud1.points = o3d.utility.Vector3dVector(c1)
    cloud2 = o3d.geometry.PointCloud()
    cloud2.points = o3d.utility.Vector3dVector(c2)
    cloud3 = o3d.geometry.PointCloud()
    cloud3.points = o3d.utility.Vector3dVector(c3)
    cloud1.paint_uniform_color([1, 0, 0])
    cloud2.paint_uniform_color([0, 1, 0])
    cloud3.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([cloud1, cloud2, cloud3],  # 待显示的点云列表
                                      window_name="点云显示",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)

def compare4(c1, c2, c3, c4):
    if c1.shape[0] == 1:
        c1 = c1[0]
        c2 = c2[0]
        c3 = c3[0]
        c4 = c4[0]
    if c1.shape[0] == 3:
        c1 = c1.T
        c2 = c2.T
        c3 = c3.T
        c4 = c4.T
    cloud1 = o3d.geometry.PointCloud()
    cloud1.points = o3d.utility.Vector3dVector(c1)
    cloud2 = o3d.geometry.PointCloud()
    cloud2.points = o3d.utility.Vector3dVector(c2)
    cloud3 = o3d.geometry.PointCloud()
    cloud3.points = o3d.utility.Vector3dVector(c3)
    cloud1.paint_uniform_color([1, 0, 0])
    cloud2.paint_uniform_color([0, 1, 0])
    cloud3.paint_uniform_color([0, 0, 1])
    cloud4 = o3d.geometry.PointCloud()
    cloud4.points = o3d.utility.Vector3dVector(c4)
    cloud4.paint_uniform_color([0, 1, 1])

    o3d.visualization.draw_geometries([cloud1, cloud2, cloud3, cloud4],  # 待显示的点云列表
                                      window_name="点云显示",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)