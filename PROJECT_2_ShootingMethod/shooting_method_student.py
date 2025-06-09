#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 学生代码模板

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

学生姓名：年世玺
学号：20231050111
完成日期：2025-06-09
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


def ode_system_shooting(t, y):
    """
    Define the ODE system for shooting method.
    
    Convert the second-order ODE u'' = -π(u+1)/4 into a first-order system:
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π(y1+1)/4
    
    Args:
        t (float): Independent variable (time/position)
        y (array): State vector [y1, y2] where y1=u, y2=u'
    
    Returns:
        list: Derivatives [y1', y2']
    """
    dydt = [
        y[1],  # y1' = y2
        -np.pi * (y[0] + 1) / 4  # y2' = -π(y1+1)/4
    ]
    return dydt


def boundary_conditions_scipy(ya, yb):
    """
    Define boundary conditions for scipy.solve_bvp.
    
    Boundary conditions: u(0) = 1, u(1) = 1
    ya[0] should equal 1, yb[0] should equal 1
    
    Args:
        ya (array): Values at left boundary [u(0), u'(0)]
        yb (array): Values at right boundary [u(1), u'(1)]
    
    Returns:
        array: Boundary condition residuals
    """
    # Residuals: [u(0)-1, u(1)-1]
    return np.array([ya[0] - 1, yb[0] - 1])


def ode_system_scipy(x, y):
    """
    Define the ODE system for scipy.solve_bvp.
    
    Note: scipy.solve_bvp uses (x, y) parameter order, different from odeint
    
    Args:
        x (float): Independent variable
        y (array): State vector [y1, y2]
    
    Returns:
        array: Derivatives as column vector
    """
    # Return column vector: [[y1'], [y2']]
    return np.vstack((
        y[1],  # y1' = y2
        -np.pi * (y[0] + 1) / 4  # y2' = -π(y1+1)/4
    ))


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    """
    Solve boundary value problem using shooting method.
    
    Algorithm:
    1. Guess initial slope m1
    2. Solve IVP with initial conditions [u(0), m1]
    3. Check if u(1) matches boundary condition
    4. If not, adjust slope using secant method and repeat
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of discretization points
        max_iterations (int): Maximum iterations for shooting
        tolerance (float): Convergence tolerance
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    """
    # Validate input parameters
    if len(x_span) != 2 or x_span[0] >= x_span[1]:
        raise ValueError("Invalid x_span. Must be (start, end) with start < end")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be tuple of length 2")
    if n_points < 3:
        raise ValueError("n_points must be at least 3")
    
    # Extract boundary conditions
    u_left, u_right = boundary_conditions
    x0, x_end = x_span
    
    # Define function to compute residual at right boundary
    def residual(m):
        """Compute residual u(x_end) - u_right for given initial slope m"""
        sol = solve_ivp(
            ode_system_shooting,
            [x0, x_end],
            [u_left, m],  # Initial conditions: [u(0), u'(0)=m]
            t_eval=np.linspace(x0, x_end, n_points)
        )
        return sol.y[0, -1] - u_right
    
    # Secant method for root finding
    m0 = 0.0  # First guess for initial slope
    m1 = -1.0  # Second guess for initial slope
    
    # Evaluate residuals
    r0 = residual(m0)
    r1 = residual(m1)
    
    # Iterate until convergence or max iterations
    iteration = 0
    while iteration < max_iterations and abs(r1) > tolerance:
        # Avoid division by zero
        if abs(r1 - r0) < 1e-12:
            break
            
        # Secant method update
        m_new = m1 - r1 * (m1 - m0) / (r1 - r0)
        
        # Update for next iteration
        m0, m1 = m1, m_new
        r0, r1 = r1, residual(m1)
        iteration += 1
    
    # Final solution with converged slope
    x = np.linspace(x0, x_end, n_points)
    sol = solve_ivp(
        ode_system_shooting,
        [x0, x_end],
        [u_left, m1],
        t_eval=x
    )
    
    return x, sol.y[0]


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    Solve boundary value problem using scipy.solve_bvp.
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of initial mesh points
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    """
    # Validate input
    if len(x_span) != 2 or x_span[0] >= x_span[1]:
        raise ValueError("Invalid x_span. Must be (start, end) with start < end")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be tuple of length 2")
    if n_points < 3:
        raise ValueError("n_points must be at least 3")
    
    # Setup initial mesh and guess
    x_mesh = np.linspace(x_span[0], x_span[1], n_points)
    
    # Initial guess: linear function satisfying boundary conditions
    u_guess = np.linspace(boundary_conditions[0], boundary_conditions[1], n_points)
    y_guess = np.zeros((2, n_points))
    y_guess[0] = u_guess  # u(x)
    y_guess[1] = 0.0      # u'(x) - initial guess zero
    
    # Solve BVP
    sol = solve_bvp(
        ode_system_scipy,
        boundary_conditions_scipy,
        x_mesh,
        y_guess,
        tol=1e-6
    )
    
    if not sol.success:
        raise RuntimeError(f"BVP solver failed: {sol.message}")
    
    # Evaluate solution on finer grid for smooth plot
    x_fine = np.linspace(x_span[0], x_span[1], 100)
    y_fine = sol.sol(x_fine)[0]
    
    return x_fine, y_fine


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    Compare shooting method and scipy.solve_bvp, generate comparison plot.
    
    Args:
        x_span (tuple): Domain for the problem
        boundary_conditions (tuple): Boundary values (left, right)
        n_points (int): Number of points for plotting
    
    Returns:
        dict: Dictionary containing solutions and analysis
    """
    # Solve using both methods
    try:
        x_shooting, y_shooting = solve_bvp_shooting_method(
            x_span, boundary_conditions, n_points)
    except Exception as e:
        print(f"Shooting method failed: {e}")
        raise
    
    try:
        x_scipy, y_scipy = solve_bvp_scipy_wrapper(
            x_span, boundary_conditions, n_points)
    except Exception as e:
        print(f"Scipy.solve_bvp failed: {e}")
        raise
    
    # Interpolate to common grid for comparison
    y_scipy_interp = np.interp(x_shooting, x_scipy, y_scipy)
    
    # Calculate differences
    diff = y_shooting - y_scipy_interp
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 8))
    
    # Plot solutions
    plt.subplot(2, 1, 1)
    plt.plot(x_shooting, y_shooting, 'b-', linewidth=2, label='Shooting Method')
    plt.plot(x_scipy, y_scipy, 'r--', linewidth=2, label='scipy.solve_bvp')
    plt.title('Comparison of Shooting Method and scipy.solve_bvp', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Plot difference
    plt.subplot(2, 1, 2)
    plt.plot(x_shooting, diff, 'g-', linewidth=2)
    plt.title('Difference Between Methods', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Difference', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('method_comparison.png')
    plt.show()
    
    # Print results
    print(f"Maximum difference: {max_diff:.4e}")
    print(f"RMS difference: {rms_diff:.4e}")
    
    # Return results dictionary
    return {
        'x_shooting': x_shooting,
        'y_shooting': y_shooting,
        'x_scipy': x_scipy,
        'y_scipy': y_scipy,
        'max_difference': max_diff,
        'rms_difference': rms_diff
    }


# Test functions for development and debugging
def test_ode_system():
    """
    Test the ODE system implementation.
    """
    print("Testing ODE system...")
    try:
        # Test point
        t_test = 0.5
        y_test = np.array([1.0, 0.5])
        
        # Test shooting method ODE system
        dydt = ode_system_shooting(t_test, y_test)
        print(f"ODE system (shooting): dydt = {dydt}")
        
        # Test scipy ODE system
        dydt_scipy = ode_system_scipy(t_test, y_test)
        print(f"ODE system (scipy): dydt = {dydt_scipy}")
        
    except Exception as e:
        print(f"ODE system test failed: {str(e)}")


def test_boundary_conditions():
    """
    Test the boundary conditions implementation.
    """
    print("Testing boundary conditions...")
    try:
        ya = np.array([1.0, 0.5])  # Left boundary
        yb = np.array([1.0, -0.3])  # Right boundary
        
        bc_residual = boundary_conditions_scipy(ya, yb)
        print(f"Boundary condition residuals: {bc_residual}")
        
    except Exception as e:
        print(f"Boundary conditions test failed: {str(e)}")


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题")
    print("=" * 50)
    
    # Run basic tests
    test_ode_system()
    test_boundary_conditions()
    
    # Run comparison
    print("\nRunning method comparison...")
    results = compare_methods_and_plot()
    print("\nComparison completed successfully!")
