"""
车载激光雷达数据适配接口
底层依赖：Common-LSG io_utils 通用IO工具
功能：解析车载LiDAR点云、街景数据，转换为标准截面格式
"""
import numpy as np
from Common_LSG.io_utils import SectionDataStandard

class VehicleLidarParser(SectionDataStandard):
    def load_lidar_pointcloud(self, pcd_path):
        """加载PCD格式车载激光点云"""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("需安装open3d依赖：pip install open3d")
        
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        section_stack = self.pointcloud_to_section_stack(points, resolution=0.1)
        standard_data = self.format_section_stack(section_stack, spacing=[0.1, 0.1, 0.1])
        return standard_data