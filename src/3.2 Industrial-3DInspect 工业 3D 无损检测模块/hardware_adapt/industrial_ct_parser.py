"""
工业CT数据适配接口
底层依赖：Common-LSG io_utils 通用IO工具
功能：解析工业CT断层数据，支持高对比度大密度差影像
"""
import numpy as np
from Common_LSG.io_utils import SectionDataStandard

class IndustrialCTParser(SectionDataStandard):
    def load_industrial_ct_series(self, ct_folder_path):
        """加载工业CT断层序列"""
        try:
            import pydicom
        except ImportError:
            raise ImportError("需安装pydicom依赖")
        
        slices = []
        import os
        for file in sorted(os.listdir(ct_folder_path)):
            if file.endswith(".dcm"):
                ds = pydicom.dcmread(os.path.join(ct_folder_path, file))
                slices.append(ds)
        
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        pixel_array = np.stack([s.pixel_array for s in slices])
        hu_array = pixel_array * slices[0].RescaleSlope + slices[0].RescaleIntercept
        
        standard_data = self.format_section_stack(
            hu_array,
            spacing=slices[0].PixelSpacing + [slices[0].SliceThickness],
            origin=slices[0].ImagePositionPatient
        )
        return standard_data