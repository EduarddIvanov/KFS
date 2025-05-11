import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lorenz_system(point, sigma, rho, beta):
    x, y, z = point
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

def integrate_lorenz(initial_point, params, dt, steps):
    sigma, rho, beta = params
    points = np.zeros((steps, 3))
    points[0] = initial_point
    for i in range(1, steps):
        k1 = lorenz_system(points[i - 1], sigma, rho, beta)
        k2 = lorenz_system(points[i - 1] + dt / 2 * k1, sigma, rho, beta)
        k3 = lorenz_system(points[i - 1] + dt / 2 * k2, sigma, rho, beta)
        k4 = lorenz_system(points[i - 1] + dt * k3, sigma, rho, beta)
        points[i] = points[i - 1] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return points

params = (10.0, 28.0, 8.0 / 3.0)
dt = 0.01
steps = 10000
initial_point1 = np.array([0.0, 1.0, 1.05])
initial_point2 = np.array([0.0, 1.0, 1.05001])
trajectory1 = integrate_lorenz(initial_point1, params, dt, steps)
trajectory2 = integrate_lorenz(initial_point2, params, dt, steps)
distances = np.linalg.norm(trajectory1 - trajectory2, axis=1)
time = np.arange(steps) * dt
fig = plt.figure(figsize=(12, 7))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(trajectory1[:, 0], trajectory1[:, 1], trajectory1[:, 2], 'b-', linewidth=0.5, label='z₀ = 1.05')
ax1.plot(trajectory2[:, 0], trajectory2[:, 1], trajectory2[:, 2], 'r-', linewidth=0.5, label='z₀ = 1.05001')
ax1.set_title('Атрактор Лоренца\n(розходження двох близьких траєкторій)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(time, distances, color='purple')
ax2.set_title('Відстань між траєкторіями з часом (log шкала)')
ax2.set_xlabel('Час')
ax2.set_ylabel('Відстань')
ax2.set_yscale('log')
ax2.grid(True)

plt.tight_layout()
plt.show()
