import matplotlib.pyplot as plt
import numpy as np
from sympy import Point, Line, Circle, sqrt, N
from sympy.geometry import Triangle
from cycleGAN.plot_geometry_problems.mat_plot_lib_2_svg import SympyGeo2SVG

svg_plotter = SympyGeo2SVG()

# Define points A, B, C
A = Point(0, 0)
svg_plotter.add_point('A', A)
B = Point(6, 0)
svg_plotter.add_point('B', B)
C = Point(2, 5)
svg_plotter.add_point('C', C)
svg_plotter.add_line(A, B)
svg_plotter.add_line(B, C)
svg_plotter.add_line(C, A)

# Compute point D on side BC using the Angle Bisector Theorem
AB = A.distance(B)
AC = A.distance(C)
D = (AC * B + AB * C) / (AB + AC)

# Compute the circumcircle of triangle ABC
circumcircle = Circle(A, B, C)

# Compute vectors along AB and AC
vector_AB = B - A
vector_AC = C - A

# Compute lengths of vectors
length_AB = sqrt(vector_AB.dot(vector_AB))
length_AC = sqrt(vector_AC.dot(vector_AC))

# Normalize the vectors
norm_vector_AB = vector_AB / length_AB
norm_vector_AC = vector_AC / length_AC

# Compute angle bisector vector
bisector_vector = norm_vector_AB + norm_vector_AC

# Define angle bisector line
angle_bisector = Line(A, A + bisector_vector)

# Find point E: intersection of angle bisector with circumcircle (other than A)
intersections = circumcircle.intersection(angle_bisector)
E_candidates = [pt for pt in intersections if not pt.equals(A)]
if E_candidates:
    E = E_candidates[0]
else:
    raise ValueError("No valid intersection point E found.")

# Compute foot F of the perpendicular from D to side AC
line_AC = Line(A, C)
perpendicular = line_AC.perpendicular_line(D)
F = line_AC.intersection(perpendicular)[0]
svg_plotter.add_line(D, F)

# Convert sympy Points to numerical coordinates for plotting
def point_to_coords(p):
    return float(N(p.x)), float(N(p.y))

A_coords = point_to_coords(A)
B_coords = point_to_coords(B)
C_coords = point_to_coords(C)
D_coords = point_to_coords(D)
E_coords = point_to_coords(E)
F_coords = point_to_coords(F)
O_coords = point_to_coords(circumcircle.center)

# Plotting
fig, ax = plt.subplots()

# Plot triangle ABC
triangle_x = [A_coords[0], B_coords[0], C_coords[0], A_coords[0]]
triangle_y = [A_coords[1], B_coords[1], C_coords[1], A_coords[1]]
ax.plot(triangle_x, triangle_y, 'k-', label='Triangle ABC')

# Plot angle bisector AD
ax.plot([A_coords[0], D_coords[0]], [A_coords[1], D_coords[1]], 'r--', label='Angle Bisector AD')
svg_plotter.add_line(A, E)

# Plot circumcircle ω
theta = np.linspace(0, 2 * np.pi, 400)
radius = float(N(circumcircle.radius))
circle_x = O_coords[0] + radius * np.cos(theta)
circle_y = O_coords[1] + radius * np.sin(theta)
ax.plot(circle_x, circle_y, 'b--', label='Circumcircle ω')
svg_plotter.add_circle(circumcircle.center, radius)

# Plot points
ax.plot(A_coords[0], A_coords[1], 'ko')
ax.plot(B_coords[0], B_coords[1], 'ko')
ax.plot(C_coords[0], C_coords[1], 'ko')
ax.plot(D_coords[0], D_coords[1], 'ko', label='Point D')
svg_plotter.add_point('D', D)
ax.plot(E_coords[0], E_coords[1], 'ko', label='Point E')
svg_plotter.add_point('E', E)
ax.plot(F_coords[0], F_coords[1], 'ko', label='Point F')
svg_plotter.add_point('F', F)

# Plot line EF
ax.plot([E_coords[0], F_coords[0]], [E_coords[1], F_coords[1]], 'c-', label='Line EF')
svg_plotter.add_line(E, F)

# Add labels at point locations without offsets
def add_point_label(ax, point_coords, label):
    ax.text(point_coords[0], point_coords[1], label, fontsize=12, ha='right', va='bottom')

add_point_label(ax, A_coords, 'A')
add_point_label(ax, B_coords, 'B')
add_point_label(ax, C_coords, 'C')
add_point_label(ax, D_coords, 'D')
add_point_label(ax, E_coords, 'E')
add_point_label(ax, F_coords, 'F')
add_point_label(ax, O_coords, 'O')  # Center of the circumcircle


svg_code = svg_plotter.get_svg_code()
with open('p2_o1_generated.svg', 'w') as f:
    f.write(svg_code)


# Set plot properties
ax.set_aspect('equal', 'box')
ax.legend()
ax.grid(True)
plt.show()
