"""
层级截面几何(LSG V2.0) - 分层树形3D重建【人体轮廓掩码过滤版】
新增功能：自动识别皮肤人体截面，剔除衣服、检查床、体外杂物，仅保留人体内部组织计算
优化：向量化极速插值、无GUI阻塞2D图、真实毫米层厚、层间平滑
"""
import numpy as np
import pydicom
import os
import pyvista as pv
from matplotlib import pyplot as plt
from scipy.ndimage import label, binary_closing, binary_fill_holes, sum as ndi_sum

# VTK CUDA显卡加速
os.environ["VTK_CUDA"] = "1"
os.environ["PYVISTA_GPU"] = "1"

# matplotlib 无GUI后台模式，杜绝窗口线程冲突
plt.switch_backend('Agg')
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 输出文件夹自动创建
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ===================== DICOM加载工具 =====================
def load_dicom_4d(dicom_dir):
    slices = []
    for f in sorted(os.listdir(dicom_dir)):
        if f.endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(dicom_dir, f), defer_size=None)
            try:
                _ = ds.pixel_array
            except Exception:
                continue
            slices.append(ds)
    if len(slices) == 0:
        raise RuntimeError("目录无有效DICOM切片")

    def get_z_pos(ds):
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        elif hasattr(ds, "SliceLocation"):
            return float(ds.SliceLocation)
        return 0.0

    valid_slices = []
    standard_shape = None
    for ds in slices:
        arr = ds.pixel_array
        if standard_shape is None:
            standard_shape = arr.shape
            valid_slices.append(ds)
        elif arr.shape == standard_shape:
            valid_slices.append(ds)
    valid_slices.sort(key=lambda x: get_z_pos(x))
    print(f"原始切片总数：{len(slices)}，有效统一尺寸切片：{len(valid_slices)}")

    # 读取真实物理层厚 mm
    if len(valid_slices)>=2:
        z0 = get_z_pos(valid_slices[0])
        z1 = get_z_pos(valid_slices[1])
        dz = abs(z1 - z0)
    else:
        dz = 1.0
    print(f"DICOM真实层厚 dz = {dz:.2f} mm，150层总解剖厚度：{dz*150:.1f} mm")

    volume = np.stack([s.pixel_array for s in valid_slices], axis=-1)
    return volume, dz

def calc_single_slice_grad(slice_arr):
    gx, gy = np.gradient(slice_arr)
    return np.sqrt(gx**2 + gy**2)

def save_slice_image(volume, mid_slice_idx):
    # 仅后台保存2D对比图，不弹出窗口
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    raw_img = volume[:, :, mid_slice_idx]
    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title('原始CT断层 / Raw CT Slice')
    axes[0].axis('off')

    g_slice = calc_single_slice_grad(raw_img)
    im = axes[1].imshow(g_slice, cmap='jet', vmin=0, vmax=60)
    axes[1].set_title('单层像素梯度 / Slice Gradient')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig("./output/ct_2d_gradient_compare.png", dpi=600, bbox_inches="tight")
    plt.savefig("./output/ct_2d_gradient_compare.pdf", bbox_inches="tight")
    plt.close(fig)
    print("2D断层对比图已后台保存至 ./output/")

def save_body_mask_check(volume, body_mask, slice_idx=None):
    """保存人体掩码校验图，查看皮肤轮廓裁剪效果"""
    H, W, Z = volume.shape
    if slice_idx is None:
        slice_idx = Z // 2
    raw_slice = volume[:, :, slice_idx]
    mask_slice = body_mask[:, :, slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_slice, cmap="gray")
    axes[0].set_title("原始CT断层")
    axes[0].axis("off")

    axes[1].imshow(mask_slice, cmap="gray")
    axes[1].set_title("人体皮肤轮廓掩码")
    axes[1].axis("off")

    overlay = raw_slice.copy()
    overlay[~mask_slice] = -1000
    axes[2].imshow(overlay, cmap="gray")
    axes[2].set_title("剔除杂物后人体区域")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("./output/body_mask_check.png", dpi=300, bbox_inches="tight")
    plt.savefig("./output/body_mask_check.pdf", bbox_inches="tight")
    plt.close(fig)
    print("人体掩码校验图已保存至 ./output/body_mask_check.png")

def extract_body_mask(volume, hu_min=-200, hu_max=200, min_area_ratio=0.01):
    """
    从CT体数据中提取人体/皮肤截面轮廓掩码，剔除体外衣物、检查床、杂物
    参数：
        volume : 3D CT体数据，shape = (H, W, Z)
        hu_min : 软组织/皮肤最低HU阈值
        hu_max : 软组织/皮肤最高HU阈值
        min_area_ratio : 最小连通域面积比例，过滤小型体外物体
    返回：
        body_mask : 二值掩码，True=人体内部，False=外部杂物
    """
    H, W, Z = volume.shape
    # 1. 软组织阈值粗提取（皮肤、肌肉区间）
    soft_tissue_mask = (volume >= hu_min) & (volume <= hu_max)
    # 2. 三维连通域分析
    labeled, num_labels = label(soft_tissue_mask)
    # 3. 统计每个连通域体素数量
    label_sizes = np.bincount(labeled.flatten())
    min_voxel_count = H * W * Z * min_area_ratio
    # 4. 仅保留体积达标区域（人体主体）
    valid_labels = []
    for i in range(1, num_labels + 1):
        if label_sizes[i] >= min_voxel_count:
            valid_labels.append(i)
    if not valid_labels:
        print("警告：未检测到明显人体区域，回退全图计算")
        return np.ones_like(volume, dtype=bool)
    # 5. 生成人体掩码
    body_mask = np.isin(labeled, valid_labels)
    # 6. 形态学闭运算，补全皮肤轮廓断裂缝隙
    body_mask = binary_closing(body_mask, structure=np.ones((3, 3, 3)))
    # 7. 填充胸腔内部空洞（肺、气管），形成完整闭合人体轮廓
    body_mask = binary_fill_holes(body_mask)
    print(f"人体轮廓掩码提取完成，有效人体连通域数量：{len(valid_labels)}")
    print(f"人体区域体素占整张CT比例：{body_mask.mean()*100:.2f}%")
    return body_mask

# ===================== 逐层树形连通域生长（叠加人体掩码过滤） =====================
def incremental_tree_reconstruction(volume, body_mask, density_thresh=100, grad_thresh=40, min_voxel=200):
    H, W, Z_total = volume.shape
    global_label_id = 1
    slice_label_map = {}
    slice_structure_cache = {}

    print("===== 开始逐层生长构建树形拓扑结构（仅人体内部运算） =====")
    for z in range(Z_total):
        print(f"处理切片 {z+1}/{Z_total}")
        slice_data = volume[:, :, z]
        slice_grad = calc_single_slice_grad(slice_data)
        slice_body_mask = body_mask[:, :, z]
        # 三重约束：密度阈值 + 梯度阈值 + 人体轮廓内部
        slice_mask = (slice_data > density_thresh) & (np.abs(slice_grad) > grad_thresh) & slice_body_mask
        if not np.any(slice_mask):
            slice_label_map[z] = np.zeros_like(slice_mask, dtype=np.int32)
            continue

        layer_label, _ = label(slice_mask)
        layer_label = layer_label.astype(np.int32)
        # 跨层结构绑定，树形连续ID继承
        if z > 0:
            prev_layer_label = slice_label_map[z-1]
            overlap_mask = (layer_label > 0) & (prev_layer_label > 0)
            if np.any(overlap_mask):
                match_prev_ids = prev_layer_label[overlap_mask]
                match_curr_ids = layer_label[overlap_mask]
                for p_id, c_id in zip(match_prev_ids, match_curr_ids):
                    layer_label[layer_label == c_id] = p_id
            new_mask = (layer_label == 0) & slice_mask
            layer_label[new_mask] = global_label_id
            global_label_id += 1
        else:
            nonzero_idx = np.nonzero(layer_label)
            layer_label[nonzero_idx] = global_label_id
            global_label_id += 1

        slice_label_map[z] = layer_label
        y_coords, x_coords = np.where(layer_label > 0)
        z_coords = np.full_like(x_coords, z)
        slice_structure_cache[z] = (x_coords, y_coords, z_coords, slice_data[y_coords, x_coords])

    all_x, all_y, all_z, all_hu = [], [], [], []
    for z in slice_structure_cache:
        x, y, z_arr, hu = slice_structure_cache[z]
        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z_arr)
        all_hu.extend(hu)

    raw_x = np.array(all_x)
    raw_y = np.array(all_y)
    raw_z = np.array(all_z)
    raw_hu = np.array(all_hu)
    print(f"分层生长完成，人体内部有效原始点云总数：{len(raw_x)}")
    return raw_x, raw_y, raw_z, raw_hu

# ===================== 向量化极速插值，无串行for循环 =====================
def smooth_z_interpolate_fast(x_raw, y_raw, z_raw, hu_raw, dz, interp_step=2, sample_ratio=0.30):
    print(f"原始点云数量：{len(x_raw)}，下采样比例 {sample_ratio*100}% 过滤细碎噪点")
    # 均匀下采样削减计算量
    sample_mask = np.random.choice([True, False], size=len(x_raw), p=[sample_ratio, 1-sample_ratio])
    x_down = x_raw[sample_mask]
    y_down = y_raw[sample_mask]
    z_down = z_raw[sample_mask]
    hu_down = hu_raw[sample_mask]
    print(f"下采样后参与插值点云：{len(x_down)}")

    # Z轴转换真实毫米物理厚度，解决模型扁平
    z_real_mm = z_down * dz
    # 按Z深度全局排序
    sort_idx = np.argsort(z_real_mm)
    x_sorted = x_down[sort_idx]
    y_sorted = y_down[sort_idx]
    z_sorted = z_real_mm[sort_idx]
    hu_sorted = hu_down[sort_idx]

    print(f"执行向量化层间积分平滑插值，每层插入 {interp_step} 个过渡采样点")
    seg_count = len(z_sorted) - 1
    t = np.linspace(0, 1, interp_step)
    t_mat = np.tile(t, (seg_count, 1))
    # 矩阵广播批量插值
    z0 = z_sorted[:-1, None]
    z1 = z_sorted[1:, None]
    x0 = x_sorted[:-1, None]
    x1 = x_sorted[1:, None]
    y0 = y_sorted[:-1, None]
    y1 = y_sorted[1:, None]
    hu0 = hu_sorted[:-1, None]
    hu1 = hu_sorted[1:, None]

    z_interp = z0 * (1 - t_mat) + z1 * t_mat
    x_interp = x0 * (1 - t_mat) + x1 * t_mat
    y_interp = y0 * (1 - t_mat) + y1 * t_mat
    hu_interp = hu0 * (1 - t_mat) + hu1 * t_mat

    final_x = x_interp.ravel()
    final_y = y_interp.ravel()
    final_z = z_interp.ravel()
    final_hu = hu_interp.ravel()
    print(f"向量化插值完成，平滑后总点云：{len(final_x)}")
    print(f"胸腔真实总厚度：{np.max(final_z)-np.min(final_z):.2f} mm")
    return final_x, final_y, final_z, final_hu

# ===================== PyVista 3D渲染 =====================
def render_3d_smooth(x_arr, y_arr, z_real_mm, hu_arr, vol_shape, dz):
    H, W, Z_total = vol_shape
    plotter = pv.Plotter(window_size=(1200, 900))
    plotter.add_title("LSG V2.0 逐层树形生长·人体轮廓过滤平滑三维重建", font_size=14)

    # 白色线框包围盒，标识人体真实解剖范围
    max_x = H
    max_y = W
    max_z = Z_total * dz
    box = pv.Box(bounds=[0, max_x, 0, max_y, 0, max_z])
    plotter.add_mesh(box, color="white", opacity=0.08, style="wireframe")

    # 人体内部解剖点云渲染
    point_stack = np.column_stack([x_arr, y_arr, z_real_mm])
    point_mesh = pv.PolyData(point_stack)
    point_mesh["HU灰度"] = hu_arr.astype(np.float32)
    plotter.add_points(
        point_mesh,
        scalars="HU灰度",
        cmap="coolwarm",
        point_size=3.0,
        opacity=0.6,
        render_points_as_spheres=True,
        clim=[100, 1800] # 压缩色标区间，骨骼红色高亮更明显
    )

    plotter.add_axes(xlabel="X 像素", ylabel="Y 像素", zlabel="Z 真实厚度(mm)")
    plotter.camera_position = "iso"
    plotter.save_graphic("./output/CT_Smooth_3D_RealThick.pdf")
    print("3D平滑重建矢量图已保存至 ./output/CT_Smooth_3D_RealThick.pdf")
    plotter.show()

# ===================== 主程序入口（修正全部缩进错误） =====================
if __name__ == '__main__':
    DICOM_PATH = "./dataset/ct_scan/"
    if not os.path.exists(DICOM_PATH):
        raise FileNotFoundError(f"DICOM路径不存在：{DICOM_PATH}")
    dcm_list = [f for f in os.listdir(DICOM_PATH) if f.endswith(".dcm")]
    if len(dcm_list) == 0:
        raise RuntimeError("文件夹内无DICOM切片")

    volume, dz = load_dicom_4d(DICOM_PATH)
    H, W, Z_total = volume.shape
    print(f"体数据尺寸 H×W×Z：{H} × {W} × {Z_total}")

    # 1. 识别人体皮肤轮廓，全局剔除衣物、检查床、体外杂物
    print("正在识别人体皮肤截面轮廓，剔除衣服、平台、杂物...")
    body_mask = extract_body_mask(
        volume,
        hu_min=-200,
        hu_max=200,
        min_area_ratio=0.01
    )
    # 人体外部全部置为空气HU=-1000，后续计算完全忽略
    volume = np.where(body_mask, volume, -1000)
    # 保存掩码校验图，直观查看裁剪效果
    save_body_mask_check(volume, body_mask, slice_idx=Z_total // 2)
    # 后台保存原始+梯度2D对比图
    save_slice_image(volume, mid_slice_idx=Z_total // 2)

    # 2. 逐层树形拓扑重建（每层强制叠加人体掩码，无体外像素参与）
    x_raw, y_raw, z_raw, hu_raw = incremental_tree_reconstruction(
        volume=volume,
        body_mask=body_mask,
        density_thresh=100,
        grad_thresh=40,
        min_voxel=200
    )

    # 3. Z轴真实厚度缩放 + 向量化层间平滑插值
    x_smooth, y_smooth, z_smooth_mm, hu_smooth = smooth_z_interpolate_fast(
        x_raw=x_raw,
        y_raw=y_raw,
        z_raw=z_raw,
        hu_raw=hu_raw,
        dz=dz,
        interp_step=2,
        sample_ratio=0.30
    )

    # 4. 启动3D立体渲染窗口
    print("正在启动RTX4060Ti三维渲染窗口...")
    render_3d_smooth(x_smooth, y_smooth, z_smooth_mm, hu_smooth, vol_shape=(H, W, Z_total), dz=dz)
