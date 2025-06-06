
"""
学生模板：双摆模拟
课程：计算物理
说明：请实现标记为 TODO 的函数。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation

# 可以在函数中使用的常量
G_CONST = 9.81  # 重力加速度 (m/s^2)
L_CONST = 0.4   # 每个摆臂的长度 (m)
M_CONST = 1.0   # 每个摆锤的质量 (kg)

def derivatives(y, t, L1, L2, m1, m2, g_param):
    """
    返回双摆状态向量y的时间导数。
    此函数将被 odeint 调用。

    参数:
        y (list 或 np.array): 当前状态向量 [theta1, omega1, theta2, omega2]。
                                theta1: 第一个摆的角度 (与垂直方向的夹角)
                                omega1: 第一个摆的角速度
                                theta2: 第二个摆的角度 (如果定义为相对于第一个摆的方向，则为相对角度；如果定义为与垂直方向的夹角，则为绝对角度 - 请仔细检查题目说明！)
                                        在此问题中，根据提供的方程，theta2 也是与垂直方向的夹角。
                                omega2: 第二个摆的角速度
        t (float): 当前时间 (odeint 需要，如果方程是自治的，则可能不使用)。
        L1 (float): 第一个摆臂的长度。
        L2 (float): 第二个摆臂的长度。
        m1 (float): 第一个摆锤的质量。
        m2 (float): 第二个摆锤的质量。
        g (float): 重力加速度。

    返回:
        list 或 np.array: 时间导数 [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]。
    
    提供的运动方程 (当 L1=L2=L, m1=m2=m 时):
    (确保使用题目描述中的这些特定方程)
    dtheta1_dt = omega1
    dtheta2_dt = omega2
    domega1_dt = - (omega1**2*np.sin(2*theta1-2*theta2) + 
                    2*omega2**2*np.sin(theta1-theta2) + 
                    (g/L) * (np.sin(theta1-2*theta2) + 3*np.sin(theta1))) / 
                   (3 - np.cos(2*theta1-2*theta2))
    domega2_dt = (4*omega1**2*np.sin(theta1-theta2) + 
                  omega2**2*np.sin(2*theta1-2*theta2) + 
                  2*(g/L) * (np.sin(2*theta1-theta2) - np.sin(theta2))) / 
                 (3 - np.cos(2*theta1-2*theta2))
    """
    theta1, omega1, theta2, omega2 = y

    # TODO: 实现导数的计算。
    # 请记住，对于此问题，根据题目设置，L1=L2=L_CONST 且 m1=m2=M_CONST。
    # 您应该在公式中使用参数 L1 作为 L。(或者如果您愿意，也可以使用 L_CONST，但使用参数是良好的编程习惯)

    dtheta1_dt = omega1
    dtheta2_dt = omega2

    # 使用文档字符串和题目描述中提供的公式计算 domega1_dt
    # 确保在公式中使用 L1 (或 L2，因为它们相等) 作为 'L'。
    # num1 = ...
    # den1 = ...
    # domega1_dt = num1 / den1

    # 使用提供的公式计算 domega2_dt
    # num2 = ...
    # den2 = ...
    # domega2_dt = num2 / den2
    # 学生代码开始区域: Start
    # 将上面占位符的零替换为 domega1_dt 和 domega2_dt 的正确计算，
    # 基于文档字符串和题目描述中的方程。
    # 使用 theta1, omega1, theta2, omega2, L1 (作为 L), 和 g。

    # domega1_dt 示例 (填充 ... 部分):
    # common_denominator = 3 - np.cos(2*theta1 - 2*theta2)
    # domega1_dt_numerator = - (omega1**2 * np.sin(2*theta1 - 2*theta2) + 
    #                           2 * omega2**2 * np.sin(theta1 - theta2) + 
    #                           (g/L1) * (np.sin(theta1 - 2*theta2) + 3*np.sin(theta1)))
    # domega1_dt = domega1_dt_numerator / common_denominator

    # domega2_dt_numerator = (4 * omega1**2 * np.sin(theta1 - theta2) + 
    #                         omega2**2 * np.sin(2*theta1 - 2*theta2) + 
    #                         2 * (g/L1) * (np.sin(2*theta1 - theta2) - np.sin(theta2)))
    # domega2_dt = domega2_dt_numerator / common_denominator 
    num1 = -omega1**2 * np.sin(2*theta1 - 2*theta2) \
           - 2 * omega2**2 * np.sin(theta1 - theta2) \
           - (g_param/L1) * (np.sin(theta1 - 2*theta2) + 3*np.sin(theta1))
    den1 = 3 - np.cos(2*theta1 - 2*theta2) 
    domega1_dt = num1 / den1
    num2 = 4 * omega1**2 * np.sin(theta1 - theta2) \
           + omega2**2 * np.sin(2*theta1 - 2*theta2) \
           + 2 * (g_param/L1) * (np.sin(2*theta1 - theta2) - np.sin(theta2))
    den2 = 3 - np.cos(2*theta1 - 2*theta2)
    domega2_dt = num2 / den2
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]
    # 学生代码结束区域: End
    
    # return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt] # 取消注释并返回结果

def solve_double_pendulum(initial_conditions, t_span, t_points, L_param=L_CONST, g_param=G_CONST):
    """
    使用 odeint 和学生实现的导数函数求解双摆的常微分方程组。

    参数:
        initial_conditions (dict): {'theta1': value, 'omega1': value, 'theta2': value, 'omega2': value}
                                   角度单位为弧度，角速度单位为 rad/s。
        t_span (tuple): (t_start, t_end) 模拟的起止时间，单位为秒。
        t_points (int): 为解生成的采样点数量。
        L_param (float): 摆臂长度 (m)。默认为 L_CONST。
        g_param (float): 重力加速度 (m/s^2)。默认为 G_CONST。

    返回:
        tuple: (t_arr, sol_arr)
               t_arr: 一维 numpy 数组，包含时间点。
               sol_arr: 二维 numpy 数组，包含每个时间点的状态 [theta1, omega1, theta2, omega2]。
    
    提示:
    1. 从 `initial_conditions` 创建初始状态向量 `y0`。
    2. 使用 `np.linspace` 从 `t_span` 和 `t_points` 创建时间数组 `t_arr`。
    3. 调用 `scipy.integrate.odeint`。确保将 `derivatives` 作为第一个参数传递，
       并且 `args=(L_param, L_param, M_CONST, M_CONST, g_param)` 因为 `derivatives` 需要 L1, L2, m1, m2, g。
       (对于此问题, L1=L2=L_param 且 m1=m2=M_CONST)。
    4. 为了根据题目要求改善能量守恒 (误差 < 1e-5 J)，您可能需要向 `odeint` 传递较小的 `rtol` 和 `atol` 参数
       (例如 1e-7 或 1e-8)。
       示例: `odeint(..., rtol=1e-7, atol=1e-7)`
    """
    # TODO: 将 initial_conditions 字典转换为列表/数组 y0 以供 odeint 使用。
    # y0 = ...

    # TODO: 创建模拟所需的时间数组。
    # t_arr = ...

    # TODO: 调用 odeint 求解微分方程。
    # 使用 derivatives。请记住在 args 中为 L1 和 L2 传递 L_param，为 m1 和 m2 传递 M_CONST。
    # sol_arr = odeint(derivatives, y0, t_arr, args=(L_param, L_param, M_CONST, M_CONST, g_param), rtol=1e-7, atol=1e-7)
    
    # 学生代码开始区域: Start
    y0 = [initial_conditions['theta1'], initial_conditions['omega1'], 
          initial_conditions['theta2'], initial_conditions['omega2']]
    t_arr = np.linspace(t_span[0], t_span[1], t_points)
    sol_arr = odeint(derivatives, y0, t_arr, args=(L_param, L_param, M_CONST, M_CONST, g_param), rtol=1e-9, atol=1e-9)
    return t_arr, sol_arr

    # 学生代码结束区域: End

def calculate_energy(sol_arr, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST):
    """
    计算双摆系统的总能量 (动能 + 势能)。

    参数:
        sol_arr (np.array):来自 odeint 的解数组。每行是 [theta1, omega1, theta2, omega2]。
        L_param (float): 摆臂长度 (m)。默认为 L_CONST。
        m_param (float): 摆锤质量 (kg)。默认为 M_CONST。
        g_param (float): 重力加速度 (m/s^2)。默认为 G_CONST。

    返回:
        np.array: 一维数组，包含每个时间点的总能量。

    公式:
    势能 (V): V = -m*g*L*(2*cos(theta1) + cos(theta2))
    动能 (T):   T = m*L^2 * (omega1^2 + 0.5*omega2^2 + omega1*omega2*cos(theta1-theta2))
    总能量 (E) = T + V
    """
    # TODO: 从 sol_arr 中提取 theta1, omega1, theta2, omega2。
    # theta1 = sol_arr[:, 0]
    # omega1 = sol_arr[:, 1]
    # ... 以此类推 theta2, omega2

    # TODO: 计算势能 (V)。
    # V = ...

    # TODO: 计算动能 (T)。
    # T = ...
    
    # 学生代码开始区域: Start
    theta1 = sol_arr[:, 0]
    omega1 = sol_arr[:, 1]
    theta2 = sol_arr[:, 2]
    omega2 = sol_arr[:, 3]
    V = -m_param * g_param * L_param * (2 * np.cos(theta1) + np.cos(theta2))
    T = m_param * L_param**2 * (omega1**2 + 0.5 * omega2**2 + omega1 * omega2 * np.cos(theta1 - theta2))
    return T + V
    # 学生代码结束区域: End


# --- 可选任务: 动画 --- (自动评分器不评分，但有助于可视化)
def animate_double_pendulum(t_arr, sol_arr, L_param=L_CONST, skip_frames=10):
    """
    (可选) 创建双摆的动画。

    参数:
        t_arr (np.array): 解的时间数组。
        sol_arr (np.array): 来自 odeint 的解数组 [theta1, omega1, theta2, omega2]。
        L_param (float): 摆臂长度 (m)。
        skip_frames (int): 为控制速度，每个动画帧跳过的解的步数。

    返回:
        matplotlib.animation.FuncAnimation: 动画对象。
    
    提示:
    1. 从 `sol_arr` 中提取 theta1 和 theta2。
    2. 使用 `skip_frames` 选择动画帧 (例如, `theta1_anim = theta1_all[::skip_frames]`)。
    3. 将极坐标 (角度) 转换为笛卡尔坐标 (x, y) 以便绘图:
       x1 = L * sin(theta1)
       y1 = -L * cos(theta1)
       x2 = x1 + L * sin(theta2)
       y2 = y1 - L * cos(theta2)
    4. 设置 Matplotlib 图形和坐标轴。使用 `ax.set_aspect('equal')` 并设置合适的 `xlim`, `ylim`。
    5. 创建一个将在每帧中更新的线对象 `line, = ax.plot([], [], 'o-', ...)`。
    6. 创建一个文本对象 `time_text = ax.text(...)` 以显示当前时间。
    7. 实现一个 `init()` 函数来初始化绘图 (将线数据设置为空)。
    8. 实现一个 `animate(i)` 函数，用于更新第 `i` 帧的线数据 (`line.set_data()`) 和时间文本。
    9. 使用 `matplotlib.animation.FuncAnimation` 创建动画。
    """
    
    # 学生代码开始区域: 可选动画 Start
    # 这部分是可选的，自动评分器不会测试。
    # 如果您选择实现它，请取消注释以下行并填写详细信息。

    theta1_all = sol_arr[:, 0]
    theta2_all = sol_arr[:, 2]

    # 为动画选择帧
    theta1_anim = theta1_all[::skip_frames]
    theta2_anim = theta2_all[::skip_frames]
    t_anim = t_arr[::skip_frames]

    # 笛卡尔坐标
    x1 = L_param * np.sin(theta1_anim)
    y1 = -L_param * np.cos(theta1_anim)
    x2 = x1 + L_param * np.sin(theta2_anim)
    y2 = y1 - L_param * np.cos(theta2_anim)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, autoscale_on=False, 
                           xlim=(-2*L_param - 0.1, 2*L_param + 0.1), 
                           ylim=(-2*L_param - 0.1, 0.1))
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('双摆动画')

    line, = ax.plot([], [], 'o-', lw=2, markersize=8, color='red')
    time_template = '时间 = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % t_anim[i])
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=len(t_anim),
                                  interval=25, blit=True, init_func=init)
    return ani
    # 学生代码结束区域: 可选动画 End


if __name__ == '__main__':
    # 初始条件 (角度单位为弧度)
    initial_conditions_rad = {
        'theta1': np.pi/2,  # 90 度
        'omega1': 0.0,
        'theta2': np.pi/2,  # 90 度
        'omega2': 0.0
    }
    t_start = 0
    t_end = 100 # 使用较短时间进行快速测试，例如 10 秒或 20 秒
                       # 题目要求 100 秒，但这对于重复测试可能较慢。
    t_points_sim= 2000 # 模拟的点数。对于 100 秒，题目建议 1000-2000 点。
                            # 为了能量守恒，可能需要更多的点或更严格的 rtol/atol。

    # --- 测试 solve_double_pendulum --- 
    
    print(f"\n尝试使用学生函数求解 ODE (时间从 {t_start}s 到 {t_end}s)...")
    t_solution, sol_solution = solve_double_pendulum(
            initial_conditions_rad, 
            (t_start, t_end), 
            t_points_sim,L_param=L_CONST, g_param=G_CONST
        )
    print("solve_double_pendulum 已执行。")
        
        # --- 测试 calculate_energy ---
       
    print("\n尝试使用学生函数计算能量...")
    energy_solution = calculate_energy(sol_solution, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST)
    print("calculate_energy 已执行。")
            
    # 为学生测试绘制能量图
    plt.figure(figsize=(10, 5))
    plt.plot(t_solution, energy_solution, label='学生计算的总能量')
    plt.xlabel('时间 (s)')
    plt.ylabel('能量 (焦耳)')
    plt.title('学生：总能量 vs. 时间')
    plt.grid(True)
    plt.legend()
            
    initial_energy= energy_solution[0]
    final_energy = energy_solution[-1]
    energy_variation= np.max(energy_solution) - np.min(energy_solution)
    print(f"学生计算的初始能量: {initial_energy:.7f} J")     
    print(f"学生计算的最后能量:   {final_energy:.7f} J")
    print(f"学生计算的最大能量变化: {energy_variation:.7e} J")
    if energy_variation < 1e-5:
        print("学生能量守恒目标 (< 1e-5 J) 在此运行中已达到。")
    else:
        print(f"学生能量守恒目标未达到。变化量: {energy_variation:.2e} J。请考虑在 odeint 中增加 t_points 或调整 rtol/atol。")
    plt.ylim(initial_energy - 5*energy_variation if energy_variation > 1e-7 else initial_energy - 1e-5, 
                     initial_energy + 5*energy_variation if energy_variation > 1e-7 else initial_energy + 1e-5)
    plt.show()

    # --- 测试 animate_double_pendulum (可选) ---
    run_animation = True # 设置为 True 以测试动画
    if run_animation:
        print("\n尝试使用学生函数创建动画...")
                # 调整 skip_frames: t_points_student / (期望的fps * 动画时长_秒)
                # 例如: 1000 点 / (25fps * 10秒动画_对应10秒真实时间) = 4。暂时使用固定的跳帧数。
        anim_obj= animate_double_pendulum(t_solution, sol_solution,L_param=L_CONST,skip_frames=max(1, t_points_sim // 200))
                
        plt.show() # 显示动画
        print("Animation display complete.")
    else:
        print("Animation skipped.")

    print("Double Pendulum Simulation finished.")

        

    

"""
给学生的提示:
1.  首先实现 `derivatives`。如果可能，用简单的输入测试它，尽管它主要通过 `odeint` 进行测试。
2.  然后实现 `solve_double_pendulum`。确保正确调用 `odeint`。
3.  接下来实现 `calculate_energy`。这对于验证模拟的正确性至关重要。
4.  绘制能量图。如果能量不守恒 (或显著漂移)，请重新检查 `derivatives` 中的方程是否有误，
    或在 `solve_double_pendulum` 的 `odeint` 调用中调整 `rtol` 和 `atol`，或增加 `t_points`。
    目标是在 100 秒内能量变化 < 1e-5 J。
5.  动画是可选的，但强烈建议用于理解物理过程。它不会被自动评分。
6.  在开发过程中，使用 `if __name__ == '__main__':` 块来测试您的函数。
    `NotImplementedError` 将在第一个未实现的函数处停止执行。
"""
