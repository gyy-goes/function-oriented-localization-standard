"""
深度图空洞修复模块（场景专属）
底层依赖：Common-LSG constraint_algo 平滑算子
功能：修复RGB-D深度图空洞、边缘噪声，提升重建连续性
"""
from Common_LSG.constraint_algo import GradientSmoother

class DepthHoleRepair(GradientSmoother):
    def depth_hole_fill(self, depth_matrix):
        """深度图空洞自适应插值修复"""
        filled_matrix = self.gradient_interpolate_hole(depth_matrix)
        return filled_matrix

    def depth_edge_smooth(self, depth_matrix):
        """深度边缘顺滑优化，消除阶梯感"""
        smooth_matrix = self.edge_gradient_smooth(depth_matrix)
        return smooth_matrix