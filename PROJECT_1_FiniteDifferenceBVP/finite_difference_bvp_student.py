#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：二阶常微分方程边值问题数值解法 - 学生代码模板

本项目要求实现两种数值方法求解边值问题：
1. 有限差分法 (Finite Difference Method)
2. scipy.integrate.solve_bvp 方法

问题设定：
y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
边界条件：y(0) = 0, y(5) = 3

学生需要完成所有标记为 TODO 的函数实现。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.linalg import solve


# ============================================================================
# 方法1：有限差分法 (Finite Difference Method)
# ============================================================================

def solve_bvp_finite_difference(n):
    """
    使用有限差分法求解二阶常微分方程边值问题。
    
    方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
    边界条件：y(0) = 0, y(5) = 3
    
    Args:
        n (int): 内部网格点数量
    
    Returns:
        tuple: (x_grid, y_solution)
            x_grid (np.ndarray): 包含边界点的完整网格
            y_solution (np.ndarray): 对应的解值
    """
    # 区间设置
    a = 0.0
    b = 5.0
    h = (b - a) / (n + 1)  # 步长
    
    # 创建网格点 (包括边界点)
    x_grid = np.linspace(a, b, n + 2)
    
    # 初始化系数矩阵A和右端向量b
    A = np.zeros((n, n))
    b_vec = np.zeros(n)
    
    # 填充系数矩阵A和右端向量b
    for i in range(n):
        x_i = x_grid[i+1]  # 内部点对应x值
        
        # 中心差分系数
        A[i, i] = -2/h**2 + np.exp(x_i)  # 主对角线：y_i项
        
        # 次对角线：y_{i-1}项
        if i > 0:
            A[i, i-1] = 1/h**2 - np.sin(x_i)/(2*h)
        
        # 超对角线：y_{i+1}项
        if i < n-1:
            A[i, i+1] = 1/h**2 + np.sin(x_i)/(2*h)
        
        # 右端项
        b_vec[i] = x_i**2
    
    # 处理边界条件对右端向量的影响
    # 左边界条件: y_0 = 0
    b_vec[0] -= (1/h**2 - np.sin(x_grid[1])/(2*h)) * 0  # y_0项
    
    # 右边界条件: y_{n+1} = 3
    b_vec[-1] -= (1/h**2 + np.sin(x_grid[-2])/(2*h)) * 3  # y_{n+1}项
    
    # 求解线性方程组
    y_internal = solve(A, b_vec)
    
    # 组合完整解（包括边界点）
    y_solution = np.zeros(n + 2)
    y_solution[0] = 0  # 左边界
    y_solution[-1] = 3  # 右边界
    y_solution[1:-1] = y_internal
    
    return x_grid, y_solution


# ============================================================================
# 方法2：scipy.integrate.solve_bvp 方法
# ============================================================================

def ode_system_for_solve_bvp(x, y):
    """
    为 scipy.integrate.solve_bvp 定义ODE系统。
    
    将二阶ODE转换为一阶系统：
    y[0] = y(x)
    y[1] = y'(x)
    
    系统方程：
    dy[0]/dx = y[1]
    dy[1]/dx = -sin(x) * y[1] - exp(x) * y[0] + x^2
    """
    y0 = y[0]  # y(x)
    y1 = y[1]  # y'(x)
    
    dy0_dx = y1
    dy1_dx = -np.sin(x) * y1 - np.exp(x) * y0 + x**2
    
    return np.vstack([dy0_dx, dy1_dx])


def boundary_conditions_for_solve_bvp(ya, yb):
    """
    为 scipy.integrate.solve_bvp 定义边界条件。
    
    左边界：y(0) = 0 → ya[0] = 0
    右边界：y(5) = 3 → yb[0] = 3
    """
    return np.array([ya[0], yb[0] - 3])


def solve_bvp_scipy(n_initial_points=11):
    """
    使用 scipy.integrate.solve_bvp 求解BVP。
    
    Args:
        n_initial_points (int): 初始网格点数
    
    Returns:
        tuple: (x_solution, y_solution)
            x_solution (np.ndarray): 解的 x 坐标数组
            y_solution (np.ndarray): 解的 y 坐标数组
    """
    # 创建初始网格
    x_initial = np.linspace(0, 5, n_initial_points)
    
    # 初始猜测：线性函数 y = (3/5)x
    y0_guess = np.linspace(0, 3, n_initial_points)
    y1_guess = np.ones(n_initial_points) * 0.6   # 常数斜率
    y_initial = np.zeros((2, n_initial_points))
    
    # 求解BVP
    sol = solve_bvp(ode_system_for_solve_bvp, boundary_conditions_for_solve_bvp, x_initial, y_initial)
    
    # 检查求解是否成功
    if not sol.success:
        raise RuntimeError(f"求解失败: {sol.message}")
    
    # 在更密集的网格上评估解以获得平滑曲线
    x_solution = sol.x
    y_solution = sol.sol(x_solution)[0]  # 只取y值，不取y'
    
    return x_solution, y_solution


# ============================================================================
# 主程序：测试和比较两种方法
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("二阶常微分方程边值问题数值解法比较")
    print("方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2")
    print("边界条件：y(0) = 0, y(5) = 3")
    print("=" * 80)
    
    # 设置参数
    x_start, y_start = 0.0, 0.0  # 左边界条件
    x_end, y_end = 5.0, 3.0      # 右边界条件
    n_points = 100  # 有限差分法的内部网格点数
    
    try:
        # 方法1：有限差分法
        print("\n1. 有限差分法求解...")
        x_fd, y_fd = solve_bvp_finite_difference(n_points - 2)
        print(f"   网格点数：{len(x_fd)}")
        print(f"   y(0) = {y_fd[0]:.6f}, y(5) = {y_fd[-1]:.6f}")
        
    except NotImplementedError:
        print("   有限差分法尚未实现")
        x_fd, y_fd = None, None
    
    try:
        # 方法2：scipy.integrate.solve_bvp
        print("\n2. scipy.integrate.solve_bvp 求解...")
        x_scipy, y_scipy = solve_bvp_scipy(n_points)
        print(f"   网格点数：{len(x_scipy)}")
        print(f"   y(0) = {y_scipy[0]:.6f}, y(5) = {y_scipy[-1]:.6f}")
        
    except NotImplementedError:
        print("   solve_bvp 方法尚未实现")
        x_scipy, y_scipy = None, None
    
    # 绘图比较
    plt.figure(figsize=(12, 8))
    
    # 子图1：解的比较
    if x_fd is not None and y_fd is not None:
        plt.plot(x_fd, y_fd, 'b-o', markersize=3, label='Finite Difference Method', linewidth=2)
    if x_scipy is not None and y_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', label='scipy.integrate.solve_bvp', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Comparison of Numerical Solutions for BVP')
    plt.legend()
    plt.grid(True, alpha=0.3)        
        # 在几个特定点比较解的值
    test_points = [1.0, 2.5, 4.0]
    
    for x_test in test_points:
        print(f"\n在 x = {x_test} 处的解值:")
        
        if x_fd is not None and y_fd is not None:
            # 插值得到测试点的值
            y_test_fd = np.interp(x_test, x_fd, y_fd)
            print(f"  有限差分法:  {y_test_fd:.6f}")
        
        if x_scipy is not None and y_scipy is not None:
            y_test_scipy = np.interp(x_test, x_scipy, y_scipy)
            print(f"  solve_bvp:   {y_test_scipy:.6f}")
    plt.tight_layout()
    plt.show()
    print("\n=" * 60)
    print("实验完成！")
    print("请在实验报告中分析两种方法的精度、效率和适用性。")
    print("=" * 60)
