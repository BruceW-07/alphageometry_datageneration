from typing import Any, Optional, Union

import numpy as np

from cycleGAN.plot_geometry_problems.mat_plot_lib_2_svg import SympyGeo2SVG
from sympy.geometry import Point

import geometry as gm
from numericals import Point, Line, Circle
def draw_svg(
    points: list[gm.Point],
    lines: list[gm.Line],
    circles: list[gm.Circle],
    segments: list[gm.Segment],
    goal: Any = None,
    highlights: list[tuple[str, list[gm.Point]]] = None,
    equals: list[tuple[Any, Any]] = None,
    theme: str = 'light',
    svg_width: int = 800,
    svg_height: int = 600,
    alpha_geo_2_org: dict = {},
    org_2_alpha_geo = None,
    goal_str = ''
) -> str:
    """Return SVG code for drawing everything on the same canvas."""

    svg_plotter = SympyGeo2SVG()

    for line in lines:
        line_points = line.neighbors(gm.Point)
        pt0 = Point(line_points[0].num.x, line_points[0].num.y)
        pt1 = Point(line_points[1].num.x, line_points[1].num.y)
        svg_plotter.add_line(pt0, pt1)

    for circle in circles:
        center = Point(circle.num.center.x, circle.num.center.y)
        svg_plotter.add_circle(center, circle.num.radius)

    # Draw points
    point_name_2_loc = {}
    for point in points:
        point_name = alpha_geo_2_org.get(point.name.upper(), point.name.upper())
        pt = Point(point.num.x, point.num.y)
        svg_plotter.add_point(point_name, pt)
        point_name_2_loc[point.name] = (pt.x, pt.y)

    draw_goal(goal_str, svg_plotter, point_name_2_loc, org_2_alpha_geo, color='green')

    return svg_plotter.get_svg_code()


def draw_perp(svg_plotter, arg_pts, color, point_name_2_loc, forward_point_name_map):
    assert len(arg_pts) == 4
    A, B, C, D = draw_pair_of_lines(svg_plotter, arg_pts, color, forward_point_name_map, point_name_2_loc)

    # Compute the intersection point of the two lines (extended if necessary)
    def line_intersection_ext(A, B, C, D):
        x1, y1 = A
        x2, y2 = B
        x3, y3 = C
        x4, y4 = D

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            # Lines are parallel
            return None

        Px_num = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        Py_num = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        Px = Px_num / denominator
        Py = Py_num / denominator
        return (Px, Py)

    intersection = line_intersection_ext(A, B, C, D)
    if intersection:
        Px, Py = intersection

        # Draw a small square at the intersection point
        # Size of the square 10% of the minimum side length
        size = min(np.linalg.norm(A - B), np.linalg.norm(C - D))/10
        # Calculate direction vectors for the two lines
        dx1, dy1 = B[0] - A[0], B[1] - A[1]
        length1 = (dx1 ** 2 + dy1 ** 2) ** 0.5
        ux1, uy1 = dx1 / length1, dy1 / length1  # Unit vector along line AB

        dx2, dy2 = D[0] - C[0], D[1] - C[1]
        length2 = (dx2 ** 2 + dy2 ** 2) ** 0.5
        ux2, uy2 = dx2 / length2, dy2 / length2  # Unit vector along line CD

        # Adjust the direction for "right up" orientation
        # First point is at the intersection
        x0, y0 = Px, Py
        # Second point along line AB (positive direction)
        x1, y1 = x0 + ux1 * size, y0 + uy1 * size
        # Third point from second point, along negative of line CD
        x2, y2 = x1 - ux2 * size, y1 - uy2 * size
        # Fourth point from intersection, along negative of line CD
        x3, y3 = x0 - ux2 * size, y0 - uy2 * size

        # Create the path for the square (no fill, with stroke)
        svg_plotter.add_line(Point(x0, y0), Point(x1, y1), clor=color)
        svg_plotter.add_line(Point(x1, y1), Point(x2, y2), clor=color)


def draw_congruent(svg_plotter, arg_pts, color, point_name_2_loc, org_2_alpha_geo):
    A, B, C, D = draw_pair_of_lines(svg_plotter, arg_pts, color, org_2_alpha_geo, point_name_2_loc)

    # Function to draw a tick mark on a line segment from P1 to P2
    def draw_tick(svg_plotter, P1, P2, color):
        # Calculate the midpoint of the line segment
        x_mid = (P1[0] + P2[0]) / 2
        y_mid = (P1[1] + P2[1]) / 2

        # Direction vector of the line segment
        dx = P2[0] - P1[0]
        dy = P2[1] - P1[1]
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length == 0:
            return ''
        # Unit vector in the direction of the line
        ux = dx / length
        uy = dy / length
        # Perpendicular vector (rotated 90 degrees)
        perp_ux = -uy
        perp_uy = ux
        # Length of the tick mark
        # tick length is 10% the line length
        tick_length = np.linalg.norm(np.array(P1) - np.array(P2)) / 10
        # Calculate the endpoints of the tick mark
        x1 = x_mid - (perp_ux * tick_length / 2)
        y1 = y_mid - (perp_uy * tick_length / 2)
        x2 = x_mid + (perp_ux * tick_length / 2)
        y2 = y_mid + (perp_uy * tick_length / 2)
        # Return the SVG line element for the tick mark
        svg_plotter.add_line(Point(x1, y1), Point(x2, y2), color=color)

    # Draw tick marks on both line segments
    draw_tick(svg_plotter, A, B, color=color)
    draw_tick(svg_plotter, C, D, color=color)



def draw_pair_of_lines(svg_plotter, arg_pts, color, org_2_alpha_geo, point_name_2_loc):
    assert len(arg_pts) == 4
    A = point_name_2_loc[org_2_alpha_geo[arg_pts[0]].lower()]
    B = point_name_2_loc[org_2_alpha_geo[arg_pts[1]].lower()]
    C = point_name_2_loc[org_2_alpha_geo[arg_pts[2]].lower()]
    D = point_name_2_loc[org_2_alpha_geo[arg_pts[3]].lower()]
    # Draw the two line segments
    svg_plotter.add_line(Point(A[0], A[1]), Point(B[0], B[1]), color=color)
    svg_plotter.add_line(Point(C[0], C[1]), Point(D[0], D[1]), color=color)
    return A, B, C, D


def draw_goal(goal_str, svg_plotter, point_name_2_loc, org_2_alpha_geo, color):
    if goal_str == '':
        return

    goal_cmd_w_args = goal_str.strip().split(' ')
    goal_cmd, arg_pts = goal_cmd_w_args[0], goal_cmd_w_args[1:]

    if goal_cmd.lower() == 'perp':
        draw_perp(svg_plotter, arg_pts, color, point_name_2_loc, org_2_alpha_geo)

    if goal_cmd.lower() == 'cong':
        draw_congruent(svg_plotter, arg_pts, color, point_name_2_loc, org_2_alpha_geo)


