"""
学生模板：平方反比引力场中的运动
文件：inverse_square_law_motion_student.py
作者：[邹远诏]
日期：[2025.6.4]

重要：函数名称、参数名称和返回值的结构必须与参考答案保持一致！
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def derivatives(t, state_vector, gm_val):
    """
    计算状态向量 [x, y, vx, vy] 的导数。
    """
    x, y, vx, vy = state_vector
    r_squared = x**2 + y**2
    r_cubed = r_squared**1.5
    
    # Avoid division by zero by adding a small epsilon if needed
    epsilon = 1e-10
    r_cubed = (r_squared + epsilon)**1.5
    
    ax = -gm_val * x / r_cubed
    ay = -gm_val * y / r_cubed
    
    return np.array([vx, vy, ax, ay])

def solve_orbit(initial_conditions, t_span, t_eval, gm_val):
    """
    使用 scipy.integrate.solve_ivp 求解轨道运动问题。
    """
    sol = solve_ivp(
        fun=derivatives,
        t_span=t_span,
        y0=initial_conditions,
        t_eval=t_eval,
        args=(gm_val,),
        method='RK45',
        rtol=1e-7,
        atol=1e-9
    )
    return sol

def calculate_energy(state_vector, gm_val, m=1.0):
    """
    计算质点的（比）机械能。
    """
    if state_vector.ndim == 1:
        x, y, vx, vy = state_vector
        r = np.sqrt(x**2 + y**2)
        v_squared = vx**2 + vy**2
        energy_per_m = 0.5 * v_squared - gm_val / r
        return energy_per_m * m
    else:
        x = state_vector[:, 0]
        y = state_vector[:, 1]
        vx = state_vector[:, 2]
        vy = state_vector[:, 3]
        r = np.sqrt(x**2 + y**2)
        v_squared = vx**2 + vy**2
        energy_per_m = 0.5 * v_squared - gm_val / r
        return energy_per_m * m

def calculate_angular_momentum(state_vector, m=1.0):
    """
    计算质点的（比）角动量 (z分量)。
    """
    if state_vector.ndim == 1:
        x, y, vx, vy = state_vector
        Lz_per_m = x * vy - y * vx
        return Lz_per_m * m
    else:
        x = state_vector[:, 0]
        y = state_vector[:, 1]
        vx = state_vector[:, 2]
        vy = state_vector[:, 3]
        Lz_per_m = x * vy - y * vx
        return Lz_per_m * m

if __name__ == "__main__":
    # Constants
    GM = 1.0
    
    # Time settings
    t_start = 0
    t_end = 20
    t_eval = np.linspace(t_start, t_end, 1000)
    
    # Task A: Different energy orbits
    # Elliptic orbit (E < 0)
    ic_ellipse = [1.0, 0.0, 0.0, 0.8]
    sol_ellipse = solve_orbit(ic_ellipse, (t_start, t_end), t_eval, GM)
    E_ellipse = calculate_energy(sol_ellipse.y.T, GM)
    
    # Parabolic orbit (E ≈ 0)
    ic_parabola = [1.0, 0.0, 0.0, np.sqrt(2)]
    sol_parabola = solve_orbit(ic_parabola, (t_start, t_end), t_eval, GM)
    E_parabola = calculate_energy(sol_parabola.y.T, GM)
    
    # Hyperbolic orbit (E > 0)
    ic_hyperbola = [1.0, 0.0, 0.0, 1.5]
    sol_hyperbola = solve_orbit(ic_hyperbola, (t_start, t_end), t_eval, GM)
    E_hyperbola = calculate_energy(sol_hyperbola.y.T, GM)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(sol_ellipse.y[0], sol_ellipse.y[1], label=f'椭圆轨道 (E={E_ellipse[0]:.2f})')
    plt.plot(sol_parabola.y[0], sol_parabola.y[1], label=f'抛物线轨道 (E={E_parabola[0]:.2f})')
    plt.plot(sol_hyperbola.y[0], sol_hyperbola.y[1], label=f'双曲线轨道 (E={E_hyperbola[0]:.2f})')
    plt.plot(0, 0, 'ro', markersize=10, label='中心天体')
    plt.title('不同能量条件下的轨道')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('energy_orbits.png')
    plt.show()
    
    # Task B: Different angular momentum orbits (fixed E < 0)
    fixed_E = -0.5
    r_p = 0.5
    v_p_values = [np.sqrt(2*(fixed_E + GM/r_p)), 
                  np.sqrt(2*(fixed_E + GM/r_p))*0.8,
                  np.sqrt(2*(fixed_E + GM/r_p))*1.2]
    
    plt.figure(figsize=(10, 8))
    for i, v_p in enumerate(v_p_values):
        ic = [r_p, 0.0, 0.0, v_p]
        sol = solve_orbit(ic, (t_start, t_end), t_eval, GM)
        Lz = calculate_angular_momentum(sol.y.T)
        plt.plot(sol.y[0], sol.y[1], label=f'轨道 {i+1}: Lz={Lz[0]:.2f}')
    
    plt.plot(0, 0, 'ro', markersize=10, label='中心天体')
    plt.title('固定能量(E=-0.5)下不同角动量的轨道')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('angular_momentum_orbits.png')
    plt.show()
