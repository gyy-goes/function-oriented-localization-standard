"""
空间平面约束模块（场景专属）
底层依赖：Common-LSG constraint_algo 基类
功能：提取空间平面层级，优化VR/AR空间注册精度，适配低算力设备
"""
from Common_LSG.constraint_algo import BaseSectionConstraint

class SpatialPlaneConstraint(BaseSectionConstraint):
    def plane_layer_extract(self, section_stack):
        """空间平面层级提取（地面、墙面、天花板）"""
        plane_layers = self.plane_segment(section_stack)
        return plane_layers

    def depth_hierarchy_optimize(self, depth_matrix, level_num=8):
        """深度层级优化，轻量化分层，降低VR渲染算力"""
        hierarchical_matrix = self.quantize_layer(depth_matrix, level_num=level_num)
        return hierarchical_matrix