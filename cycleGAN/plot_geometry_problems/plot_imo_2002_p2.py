import numpy as np
import matplotlib.pyplot as plt


def midpoint(A, B):
    return (A + B) / 2


def intersection_of_lines(P1, P2, Q1, Q2):
    # Compute the intersection of two lines defined by P1P2 and Q1Q2
    A = np.array([[P2[0] - P1[0], Q1[0] - Q2[0]], [P2[1] - P1[1], Q1[1] - Q2[1]]])
    B = np.array([Q1[0] - P1[0], Q1[1] - P1[1]])

    try:
        t, u = np.linalg.solve(A, B)
        intersection = P1 + t * (P2 - P1)
        return intersection
    except np.linalg.LinAlgError:
        return None


def perpendicular_bisector(A, B):
    # Returns midpoint and slope of the perpendicular bisector of segment AB
    midpoint = (A + B) / 2
    slope = -(B[0] - A[0]) / (B[1] - A[1])
    return midpoint, slope


def line_through_point_with_slope(P, slope, x_range):
    # Returns two points on the line passing through point P with the given slope
    x1, x2 = x_range
    y1 = slope * (x1 - P[0]) + P[1]
    y2 = slope * (x2 - P[0]) + P[1]
    return np.array([x1, y1]), np.array([x2, y2])


# Define the center O and radius of the circle
O = np.array([0, 0])
r = 5

# Define points B and C (BC is the diameter of the circle)
B = np.array([-r, 0])
C = np.array([r, 0])

# Define point A on the circle such that 0 < ∠AOB < 120 degrees
theta_A = np.radians(60)
A = np.array([r * np.cos(theta_A), r * np.sin(theta_A)])

# Define point D, the midpoint of arc AB not containing C
theta_D = (theta_A + np.pi) / 2
D = np.array([r * np.cos(theta_D), r * np.sin(theta_D)])

# Define line L, which passes through O and is parallel to AD
slope_AD = (D[1] - A[1]) / (D[0] - A[0])
slope_L = slope_AD  # Parallel to AD
P1_L, P2_L = line_through_point_with_slope(O, slope_L, x_range=[-r, r])

# Find the intersection of line L with line AC
J = intersection_of_lines(P1_L, P2_L, A, C)

# Perpendicular bisector of segment OA
mid_OA, slope_perp_OA = perpendicular_bisector(O, A)
P1_perp_OA, P2_perp_OA = line_through_point_with_slope(mid_OA, slope_perp_OA, x_range=[-r, r])

# Find points E and F where the perpendicular bisector intersects the circle
t = np.linspace(0, 2 * np.pi, 1000)
x_circle = r * np.cos(t)
y_circle = r * np.sin(t)


# Intersection points of the perpendicular bisector with the circle (E and F)
def circle_equation(x, r):
    return np.sqrt(r ** 2 - x ** 2)


E = np.array([mid_OA[0], circle_equation(mid_OA[0], r)])
F = np.array([mid_OA[0], -circle_equation(mid_OA[0], r)])

# Plot the circle
plt.figure(figsize=(8, 8))
plt.plot(x_circle, y_circle, 'k-', label='Circle ω')

# Plot points A, B, C, D, O
plt.plot(A[0], A[1], 'ro', label='Point A')
plt.plot(B[0], B[1], 'bo', label='Point B')
plt.plot(C[0], C[1], 'go', label='Point C')
plt.plot(O[0], O[1], 'ko', label='Center O')
plt.plot(D[0], D[1], 'mo', label='Point D')

# Plot the lines AC and AD
plt.plot([A[0], C[0]], [A[1], C[1]], 'g--', label='Line AC')
plt.plot([A[0], D[0]], [A[1], D[1]], 'r--', label='Line AD')

# Plot line L (parallel to AD and passing through O)
plt.plot([P1_L[0], P2_L[0]], [P1_L[1], P2_L[1]], 'm-', label='Line L (Parallel to AD)')

# Plot the perpendicular bisector of OA
plt.plot([P1_perp_OA[0], P2_perp_OA[0]], [P1_perp_OA[1], P2_perp_OA[1]], 'b--', label='Perpendicular Bisector of OA')

# Plot points E, F, and J
plt.plot(E[0], E[1], 'co', label='Point E')
plt.plot(F[0], F[1], 'co', label='Point F')
plt.plot(J[0], J[1], 'yo', label='Point J')

# Annotate the plot
plt.annotate('A', A, textcoords="offset points", xytext=(-10, -10), ha='center')
plt.annotate('B', B, textcoords="offset points", xytext=(-10, -10), ha='center')
plt.annotate('C', C, textcoords="offset points", xytext=(10, -10), ha='center')
plt.annotate('O', O, textcoords="offset points", xytext=(10, -10), ha='center')
plt.annotate('D', D, textcoords="offset points", xytext=(10, -10), ha='center')
plt.annotate('E', E, textcoords="offset points", xytext=(10, -10), ha='center')
plt.annotate('F', F, textcoords="offset points", xytext=(10, -10), ha='center')
plt.annotate('J', J, textcoords="offset points", xytext=(10, -10), ha='center')

# Plot settings
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Geometry Problem Visualization')
plt.show()
