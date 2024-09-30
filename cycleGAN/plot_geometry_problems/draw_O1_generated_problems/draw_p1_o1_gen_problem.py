import matplotlib.pyplot as plt
from sympy.geometry import Point, Circle, Triangle, Line
from cycleGAN.plot_geometry_problems.mat_plot_lib_2_svg import SympyGeo2SVG

svg_plotter = SympyGeo2SVG()
# Geometry computation
# Define an acute-angled triangle ABC
A = Point(0, 0)
B = Point(6, 0)
C = Point(2, 5)
triangle = Triangle(A, B, C)

# Compute the circumcircle ω
circumcircle = triangle.circumcircle

# Compute the orthocenter H
H = triangle.orthocenter

# Compute the sides of the triangle
side_BC = Line(B, C)
side_AC = Line(A, C)
side_AB = Line(A, B)

# Compute the altitudes from A, B, and C
altitude_A = side_BC.perpendicular_line(A)
altitude_B = side_AC.perpendicular_line(B)
altitude_C = side_AB.perpendicular_line(C)

# Function to find the other intersection point of a line and a circle
def other_circle_line_intersection(circle, line, vertex):
    intersections = circle.intersection(line)
    for point in intersections:
        if not point.equals(vertex):
            return point
    return None

# Find points D, E, and F
D = other_circle_line_intersection(circumcircle, altitude_A, A)
E = other_circle_line_intersection(circumcircle, altitude_B, B)
F = other_circle_line_intersection(circumcircle, altitude_C, C)

# Compute reflections of H over sides BC, AC, and AB
H_reflect_BC = H.reflect(side_BC)
H_reflect_AC = H.reflect(side_AC)
H_reflect_AB = H.reflect(side_AB)

# Prepare for plotting
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')

# Plot triangle ABC
def plot_line(p1, p2, **kwargs):
    x_values = [float(p1.x), float(p2.x)]
    y_values = [float(p1.y), float(p2.y)]
    ax.plot(x_values, y_values, **kwargs)

plot_line(A, B, color='black')
plot_line(B, C, color='black')
plot_line(C, A, color='black')

svg_plotter.add_line(A, B)
svg_plotter.add_line(B, C)
svg_plotter.add_line(C, A)


# Plot circumcircle ω
center = circumcircle.center
radius = float(circumcircle.radius)
circle = plt.Circle((float(center.x), float(center.y)), radius, fill=False, color='blue')
ax.add_artist(circle)

svg_plotter.add_circle(center, radius)

# Plot altitudes
# Altitude from A
alt_A_intersections = altitude_A.intersection(circumcircle)
alt_A_points = [A] + [pt for pt in alt_A_intersections if not pt.equals(A)]
if len(alt_A_points) == 2:
    plot_line(alt_A_points[0], alt_A_points[1], color='green', linestyle='--')
    svg_plotter.add_line(alt_A_points[0], alt_A_points[1])

# Altitude from B
alt_B_intersections = altitude_B.intersection(circumcircle)
alt_B_points = [B] + [pt for pt in alt_B_intersections if not pt.equals(B)]
if len(alt_B_points) == 2:
    plot_line(alt_B_points[0], alt_B_points[1], color='green', linestyle='--')
    svg_plotter.add_line(alt_B_points[0], alt_B_points[1])

# Altitude from C
alt_C_intersections = altitude_C.intersection(circumcircle)
alt_C_points = [C] + [pt for pt in alt_C_intersections if not pt.equals(C)]
if len(alt_C_points) == 2:
    plot_line(alt_C_points[0], alt_C_points[1], color='green', linestyle='--')
    svg_plotter.add_line(alt_C_points[0], alt_C_points[1])


# Plot points A, B, C, H, D, E, F
def plot_point(p, label, color='black'):
    ax.plot(float(p.x), float(p.y), 'o', color=color)
    ax.text(float(p.x) + 0.1, float(p.y) + 0.1, label, color=color)

plot_point(A, 'A')
svg_plotter.add_point('A', A)
plot_point(B, 'B')
svg_plotter.add_point('B', B)
plot_point(C, 'C')
svg_plotter.add_point('C', C)
plot_point(H, 'H', color='red')
svg_plotter.add_point('H', H)
plot_point(D, 'D', color='orange')
svg_plotter.add_point('D', D)
plot_point(E, 'E', color='orange')
svg_plotter.add_point('E', E)
plot_point(F, 'F', color='orange')
svg_plotter.add_point('F', F)

svg_code = svg_plotter.get_svg_code()
with open('p1_o1_generated.svg', 'w') as f:
    f.write(svg_code)

# Plot reflections of H over sides
plot_point(H_reflect_BC, "H'", color='purple')
plot_point(H_reflect_AC, "H''", color='purple')
plot_point(H_reflect_AB, "H'''", color='purple')

# Set plot limits
all_x = [float(p.x) for p in [A, B, C, D, E, F, H, H_reflect_BC, H_reflect_AC, H_reflect_AB]]
all_y = [float(p.y) for p in [A, B, C, D, E, F, H, H_reflect_BC, H_reflect_AC, H_reflect_AB]]
ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
ax.set_ylim(min(all_y) - 1, max(all_y) + 1)

# Show grid
ax.grid(True, which='both')

# Show plot
plt.title('Geometry Problem Illustration')
plt.show()
