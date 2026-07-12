# Medical-CT 医学CT三维重建模块
## 模块定位
基于Common-LSG通用几何内核的医学CT场景专属优化模块，支持任意角度非正交截面重建、靶器官精准分割、术前三维建模。

## 核心功能
1. 标准DICOM CT数据解析，自动转换HU值
2. 金属伪影、射束硬化伪影专属消除算法
3. 人体多组织HU值边界约束，提升重建分层精度
4. 对接Common-LSG非正交重建内核，输出顺滑三维体模型

## 依赖
- 底层内核：Common-LSG v0.1+
- 第三方依赖：numpy, pydicom


## CT DICOM 3D重建运行教程
## 1.环境准备
1.安装Git+Python3.10
2.打开仓库根目录Git Bash，执行 pip install -r requirements.txt
## 2.数据集存放规范
在仓库根目录新建dataset/ct_scan，所有dcm切片直接放入ct_scan文件夹
## 3.可调参数
CHUNK_SIZE：分块读取切片数量，低配电脑设20；
DOWNSAMPLE_STEP：分辨率采样间隔；
PREVIEW_OPEN：False关闭实时预览，降低显存占用
## 4.运行命令
python main.py
## 5.结果输出
三维STL/OBJ模型、运算指标保存在./output文件夹
## 合规说明
仅学术使用，商用需确认CT数据集授权


## 调用示例
```python
from Medical_CT.hardware_adapt.dicom_ct_parser import DicomCTParser
from Common_LSG.slice_rebuild import SectionRebuilder

parser = DicomCTParser()
ct_data = parser.load_dicom_series("./dicom_data")
rebuilder = SectionRebuilder()
model = rebuilder.non_orthogonal_rebuild(ct_data.section_stack, ct_data.spacing)
开源声明
本模块场景优化代码遵循 MIT 协议，底层几何内核永久归入人类公共知识领域，禁止基于本模块核心逻辑申请独占专利。

