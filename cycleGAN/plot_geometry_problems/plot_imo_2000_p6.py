import numpy as np
import matplotlib.pyplot as plt

def foot_of_perpendicular(P, A, B):
    # Compute the foot of the perpendicular from point P to line AB
    # Line AB is defined by points A and B
    # Return point F

    AP = P - A
    AB = B - A
    t = np.dot(AP, AB) / np.dot(AB, AB)
    F = A + t * AB
    return F

def reflect_point_over_line(P, A, B):
    # Reflect point P over line AB
    # Return reflected point P_reflected
    # Line AB is defined by points A and B

    # Direction vector of the line
    D = B - A
    D = D / np.linalg.norm(D)

    # Compute normal vector to the line
    n = np.array([-D[1], D[0]])

    # Compute vector from A to P
    AP = P - A

    # Compute projection of AP onto n
    distance = np.dot(AP, n)

    # Compute reflected point
    P_reflected = P - 2 * distance * n

    return P_reflected

def reflect_line_over_line(P1, P2, A, B):
    # Reflect line defined by points P1 and P2 over line AB
    # Return reflected points P1_reflected and P2_reflected
    P1_reflected = reflect_point_over_line(P1, A, B)
    P2_reflected = reflect_point_over_line(P2, A, B)
    return P1_reflected, P2_reflected

def line_intersection(P1, P2, Q1, Q2):
    # Compute intersection point of lines P1P2 and Q1Q2
    # Return point of intersection or None if lines are parallel

    # Line P1P2 represented as a1x + b1y = c1
    a1 = P2[1] - P1[1]
    b1 = P1[0] - P2[0]
    c1 = a1*P1[0] + b1*P1[1]

    # Line Q1Q2 represented as a2x + b2y = c2
    a2 = Q2[1] - Q1[1]
    b2 = Q1[0] - Q2[0]
    c2 = a2*Q1[0] + b2*Q1[1]

    determinant = a1*b2 - a2*b1

    if abs(determinant) < 1e-10:
        # Lines are parallel
        return None
    else:
        x = (b2*c1 - b1*c2)/determinant
        y = (a1*c2 - a2*c1)/determinant
        return np.array([x, y])

# Define triangle ABC
A = np.array([0, 0])
B = np.array([5, 0])
C = np.array([2, 4])

# Compute side lengths
AB = np.linalg.norm(B - A)
AC = np.linalg.norm(C - A)
BC = np.linalg.norm(C - B)

# Compute incenter I
I_x = (BC*A[0] + AC*B[0] + AB*C[0]) / (AB + AC + BC)
I_y = (BC*A[1] + AC*B[1] + AB*C[1]) / (AB + AC + BC)
I = np.array([I_x, I_y])

# Compute area
Area = 0.5 * abs(np.cross(B - A, C - A))

# Compute inradius
s = (AB + AC + BC)/2
r = Area / s

# Compute points T1, T2, T3
T1 = foot_of_perpendicular(I, B, C)
T2 = foot_of_perpendicular(I, C, A)
T3 = foot_of_perpendicular(I, A, B)

# Compute points H1, H2, H3
H1 = foot_of_perpendicular(A, B, C)
H2 = foot_of_perpendicular(B, C, A)
H3 = foot_of_perpendicular(C, A, B)

# Compute the reflections of lines H1H2, H2H3, H3H1 over lines T1T2, T2T3, T3T1
# Reflect line H1H2 over line T1T2
H1H2_reflected_P1, H1H2_reflected_P2 = reflect_line_over_line(H1, H2, T1, T2)

# Reflect line H2H3 over line T2T3
H2H3_reflected_P1, H2H3_reflected_P2 = reflect_line_over_line(H2, H3, T2, T3)

# Reflect line H3H1 over line T3T1
H3H1_reflected_P1, H3H1_reflected_P2 = reflect_line_over_line(H3, H1, T3, T1)

# Compute the intersection points of the reflected lines
V1 = line_intersection(H1H2_reflected_P1, H1H2_reflected_P2, H2H3_reflected_P1, H2H3_reflected_P2)
V2 = line_intersection(H2H3_reflected_P1, H2H3_reflected_P2, H3H1_reflected_P1, H3H1_reflected_P2)
V3 = line_intersection(H3H1_reflected_P1, H3H1_reflected_P2, H1H2_reflected_P1, H1H2_reflected_P2)

# Plot the triangle ABC
plt.figure(figsize=(8,8))
plt.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'k-', label='Triangle ABC')

# Plot the incenter and incircle
circle = plt.Circle((I[0], I[1]), r, color='b', fill=False, label='Incircle ω')
plt.gca().add_artist(circle)
plt.plot(I[0], I[1], 'bo', label='Incenter I')

# Plot the points T1, T2, T3
plt.plot([T1[0], T2[0], T3[0], T1[0]], [T1[1], T2[1], T3[1], T1[1]], 'g--', label='Triangle T1T2T3')

# Plot the altitudes and their feet H1, H2, H3
plt.plot([A[0], H1[0]], [A[1], H1[1]], 'r--')
plt.plot([B[0], H2[0]], [B[1], H2[1]], 'r--')
plt.plot([C[0], H3[0]], [C[1], H3[1]], 'r--')
plt.plot([H1[0], H2[0], H3[0]], [H1[1], H2[1], H3[1]], 'ro', label='Feet of Altitudes H1, H2, H3')

# Plot the reflected lines
plt.plot([H1H2_reflected_P1[0], H1H2_reflected_P2[0]], [H1H2_reflected_P1[1], H1H2_reflected_P2[1]], 'm-', label='Reflected H1H2')
plt.plot([H2H3_reflected_P1[0], H2H3_reflected_P2[0]], [H2H3_reflected_P1[1], H2H3_reflected_P2[1]], 'm-', label='Reflected H2H3')
plt.plot([H3H1_reflected_P1[0], H3H1_reflected_P2[0]], [H3H1_reflected_P1[1], H3H1_reflected_P2[1]], 'm-', label='Reflected H3H1')

# Plot the triangle formed by the intersection points V1, V2, V3
plt.plot([V1[0], V2[0], V3[0], V1[0]], [V1[1], V2[1], V3[1], V1[1]], 'c-', label='Triangle V1V2V3')

# Plot the points V1, V2, V3
plt.plot([V1[0], V2[0], V3[0]], [V1[1], V2[1], V3[1]], 'co', label='Vertices on ω')

# Verify that V1, V2, V3 lie on the incircle
d1 = np.linalg.norm(V1 - I)
d2 = np.linalg.norm(V2 - I)
d3 = np.linalg.norm(V3 - I)

print("Distances from incenter I to V1, V2, V3:")
print("d1 = {:.4f}, d2 = {:.4f}, d3 = {:.4f}, r = {:.4f}".format(d1, d2, d3, r))

# Adjust plot
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.title('Geometry Problem Visualization')
plt.show()
