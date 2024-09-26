import matplotlib.pyplot as plt
import numpy as np

def plot_circle(ax, center, radius):
    theta = np.linspace(0, 2 * np.pi, 300)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, color='black')

# Create a new figure
fig, ax = plt.subplots()

# Define circle centers and radius
r = 1
center_G1 = (0, 0)
center_G2 = (1.5, 0)

# Plot circles G1 and G2
plot_circle(ax, center_G1, r)
plot_circle(ax, center_G2, r)

# Points of intersection M and N
M_x = 0.75
M_y = np.sqrt(r**2 - (M_x - center_G1[0])**2)
M = (M_x, M_y)
N = (M_x, -M_y)

# Points A and B where AB touches the circles (tangent points)
A = (center_G1[0], center_G1[1] + r)
B = (center_G2[0], center_G2[1] + r)

# Plot line AB (tangent to both circles)
ax.plot([A[0], B[0]], [A[1], B[1]], color='gray', linestyle='--')

# Line CD parallel to AB and passing through M
C = (center_G1[0] - M_x, M_y)
D = (center_G2[0] + M_x, M_y)
ax.plot([C[0], D[0]], [C[1], D[1]], color='gray', linestyle='--')

# Lines AC and BD
ax.plot([A[0], C[0]], [A[1], C[1]], color='blue')
ax.plot([B[0], D[0]], [B[1], D[1]], color='blue')

# Intersection point E of lines AC and BD
# Since lines AC and BD are symmetrical, E will be at x = M_x
E_x = M_x
E_y = A[1] + (C[1] - A[1]) * (E_x - A[0]) / (C[0] - A[0])
E = (E_x, E_y)

# Lines AN and BN
ax.plot([A[0], N[0]], [A[1], N[1]], color='green')
ax.plot([B[0], N[0]], [B[1], N[1]], color='green')

# Points P and Q where AN and BN meet CD
# For line AN
m_AN = (N[1] - A[1]) / (N[0] - A[0])
P_x = (M_y - A[1] + m_AN * A[0]) / m_AN
P = (P_x, M_y)

# For line BN
m_BN = (N[1] - B[1]) / (N[0] - B[0])
Q_x = (M_y - B[1] + m_BN * B[0]) / m_BN
Q = (Q_x, M_y)

# Plotting points
points = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'M': M, 'N': N, 'P': P, 'Q': Q}
for label, point in points.items():
    ax.plot(point[0], point[1], 'ko')  # Black circle marker
    ax.text(point[0] + 0.05, point[1] + 0.05, label, fontsize=12)

# Lines EP and EQ
ax.plot([E[0], P[0]], [E[1], P[1]], color='red')
ax.plot([E[0], Q[0]], [E[1], Q[1]], color='red')

# Setting the aspect ratio and limits
ax.set_aspect('equal')
ax.set_xlim(-1.5, 3)
ax.set_ylim(-1.5, 2)

# Show the plot
plt.title('Geometry Problem Illustration')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
