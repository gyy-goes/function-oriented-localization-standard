"""
共聚焦显微镜数据适配接口
底层依赖：Common-LSG io_utils 通用IO工具
功能：解析显微切片序列、共聚焦三维扫描数据
"""
import numpy as np
from Common_LSG.io_utils import SectionDataStandard

class ConfocalMicroscopeParser(SectionDataStandard):
    def load_tiff_stack(self, tiff_folder_path):
        """加载TIFF格式显微切片序列"""
        try:
            import tifffile
        except ImportError:
            raise ImportError("需安装tifffile依赖：pip install tifffile")
        
        import os
        slices = []
        for file in sorted(os.listdir(tiff_folder_path)):
            if file.endswith(".tif") or file.endswith(".tiff"):
                img = tifffile.imread(os.path.join(tiff_folder_path, file))
                slices.append(img)
        
        section_stack = np.stack(slices)
        standard_data = self.format_section_stack(
            section_stack,
            spacing=[0.0001, 0.0001, 0.0001],
            origin=[0, 0, 0]
        )
        return standard_data