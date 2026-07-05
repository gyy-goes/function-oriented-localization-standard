"""
GDE-LSG 求解器单元测试 + 示例运行
验证：算子构造正常、演化守恒性、无数值发散
"""
import numpy as np
from gde_lsg_solver import GDE_LSG_Solver


def test_solver_basic():
    solver = GDE_LSG_Solver(nx=16, ny=16, nz=16, n_layers=4)
    Phi0 = solver.init_gaussian_pulse(sigma=2.0)
    traj = solver.evolve(Phi0, dt=0.01, steps=50)

    # 基础校验：形状正确、无非数值
    assert traj.shape == (51, solver.n_total)
    assert not np.any(np.isnan(traj))
    assert not np.any(np.isinf(traj))

    # 总量近似守恒（验证空间本底公理）
    total_0 = np.sum(np.abs(traj[0]))
    total_end = np.sum(np.abs(traj[-1]))
    relative_error = abs(total_end - total_0) / total_0
    assert relative_error < 0.01, f"总量守恒偏差过大：{relative_error:.4f}"

    print("✅ 基础测试通过，总量守恒偏差：{:.4%}".format(relative_error))


if __name__ == "__main__":
    test_solver_basic()