"""
细胞层级约束模块（场景专属）
底层依赖：Common-LSG constraint_algo 基类
功能：微小尺度细胞、组织层级提取，强化亚微米级分辨率
"""
from Common_LSG.constraint_algo import BaseSectionConstraint

class CellLayerConstraint(BaseSectionConstraint):
    def cell_boundary_enhance(self, section_matrix, min_cell_size=0.5):
        """细胞边界亚像素增强，提升微小结构检出率"""
        enhanced_matrix = self.subpixel_edge_enhance(section_matrix, min_size=min_cell_size)
        return enhanced_matrix

    def fluorescence_layer_extract(self, section_stack, wavelength_channel):
        """荧光通道层级提取，分离不同标记生物结构"""
        channel_layer = self.wavelength_layer_segment(section_stack, wavelength_channel)
        return channel_layer