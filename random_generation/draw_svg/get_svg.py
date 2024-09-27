from typing import Any, Optional, Union

from pygments.lexer import default

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
    # Collect all coordinate points for scaling
    all_x = []
    all_y = []
    for point in points:
        all_x.append(point.num.x)
        all_y.append(point.num.y)

    for line in lines:
        line_points = line.neighbors(gm.Point)
        for p in line_points:
            all_x.append(p.num.x)
            all_y.append(p.num.y)
    for circle in circles:
        all_x.append(circle.num.center.x)
        all_y.append(circle.num.center.y)
        all_x.append(circle.num.center.x + circle.num.radius)
        all_x.append(circle.num.center.x - circle.num.radius)
        all_y.append(circle.num.center.y + circle.num.radius)
        all_y.append(circle.num.center.y - circle.num.radius)

    # Compute bounding box
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width = max_x - min_x
    height = max_y - min_y

    # Scaling factors to fit SVG viewport
    # Use the same scaling factor for x and y to maintain aspect ratio
    scale = min(svg_width / width if width != 0 else 1, svg_height / height if height != 0 else 1) * 0.9  # Add padding

    # Offsets to center the drawing
    offset_x = (svg_width - (width * scale)) / 2 - min_x * scale
    offset_y = (svg_height - (height * scale)) / 2 - min_y * scale

    # Function to transform coordinates
    def transform(p: Point) -> Point:
        # Flip y-axis to have origin at bottom-left
        x = p.x * scale + offset_x
        y = svg_height - (p.y * scale + offset_y)
        return Point(x, y)

    # Collect SVG snippets
    svg_elements = []

    # Add a white background rectangle
    svg_elements.append(f'<rect width="{svg_width}" height="{svg_height}" fill="white" />')

    # Draw lines
    for line in lines:
        svg_code = draw_line_element_svg(line, transform, color='black')
        if svg_code:
            svg_elements.append(svg_code)

    # Draw circles
    for circle in circles:
        svg_code = draw_circle_svg(circle, transform, scale, color='blue')
        if svg_code:
            svg_elements.append(svg_code)

    # Draw segments (optional, if you have segments to draw)
    # for segment in segments:
    #     svg_code = draw_segment_svg(segment, transform, color='green')
    #     if svg_code:
    #         svg_elements.append(svg_code)

    # Draw points
    point_name_2_loc = {}
    for point in points:
        tp = transform(point.num)
        svg_code = draw_point_svg(
            tp,
            point.name,
            color='red',
            rename_map=alpha_geo_2_org
        )
        svg_elements.append(svg_code)
        point_name_2_loc[point.name] = (tp.x, tp.y)

    svg_elements.append(draw_goal(goal_str, point_name_2_loc, org_2_alpha_geo, color='green'))

    # Combine all SVG elements into a single SVG content
    svg_content = '\n'.join(svg_elements)

    # Create the full SVG code
    svg_header = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{svg_width}" height="{svg_height}">
'''
    svg_footer = '</svg>'

    svg_code = svg_header + svg_content + svg_footer

    return svg_code


def draw_point_svg(
    tp: Point,
    name: str,
    color: str = 'black',
    size: float = 5,
    rename_map = {}
) -> str:
    """Return SVG code for drawing a point."""
    # Generate the circle element
    svg_code = f'<circle cx="{tp.x}" cy="{tp.y}" r="{size}" fill="{color}" />\n'

    # Determine position for the label
    x_label, y_label = tp.x + 10, tp.y - 10  # Simple offset for label

    # Add the text element
    point_name = rename_map.get(name.upper(), name.upper())
    svg_code += f'<text x="{x_label}" y="{y_label}" fill="{color}" font-size="{size * 2}">{point_name}</text>\n'

    return svg_code

def draw_line_svg(
    p1: Point,
    p2: Point,
    transform,
    color: str = 'black',
    lw: float = 2,
    alpha: float = 0.8,
) -> str:
    """Return SVG code for drawing a line between two points."""
    # Transform the point coordinates
    tp1 = transform(p1)
    tp2 = transform(p2)

    # Generate SVG line element
    svg_code = f'<line x1="{tp1.x}" y1="{tp1.y}" x2="{tp2.x}" y2="{tp2.y}" stroke="{color}" stroke-width="{lw}" opacity="{alpha}" />\n'
    return svg_code

def draw_line_element_svg(
    line: Line,
    transform,
    color: str = 'black',
    lw: float = 2,
    alpha: float = 0.8,
) -> str:
    """Return SVG code for drawing a line."""
    points = line.neighbors(gm.Point)
    if len(points) <= 1:
        return ''
    points = [p.num for p in points]
    p1, p2 = points[:2]
    return draw_line_svg(p1, p2, transform, color, lw, alpha)

def draw_circle_svg(
    circle: Circle,
    transform,
    scale: float,
    color: str = 'cyan',
    lw: float = 2,
) -> str:
    """Return SVG code for drawing a circle."""
    # Ensure we have a numerical circle
    if circle.num is not None:
        circle = circle.num
    else:
        points = circle.neighbors(gm.Point)
        if len(points) <= 2:
            return ''
        points = [p.num for p in points]
        p1, p2, p3 = points[:3]
        circle = Circle(p1=p1, p2=p2, p3=p3)

    # Transform center
    center = transform(circle.center)

    # Scale radius
    radius = circle.radius * scale

    # Generate SVG circle element
    svg_code = (f'<circle cx="{center.x}" cy="{center.y}" r="{radius}" stroke="{color}" '
                f'stroke-width="{lw}" fill="none" />\n')
    return svg_code


def draw_perp(arg_pts, color, point_name_2_loc, forward_point_name_map):
    assert len(arg_pts) == 4
    A, B, C, D, svg_cmd = draw_pair_of_lines(arg_pts, color, forward_point_name_map, point_name_2_loc)

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
        size = 10  # Size of the square
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
        square_path = f'M {x0} {y0} L {x1} {y1} L {x2} {y2} L {x3} {y3} Z'
        svg_cmd += f'<path d="{square_path}" fill="none" stroke="{color}" stroke-width="2"/>\n'
    return svg_cmd


def draw_congruent(arg_pts, color, point_name_2_loc, org_2_alpha_geo):
    A, B, C, D, svg_cmd = draw_pair_of_lines(arg_pts, color, org_2_alpha_geo, point_name_2_loc)

    # Function to draw a tick mark on a line segment from P1 to P2
    def draw_tick(P1, P2):
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
        tick_length = 10  # Adjust this value as needed
        # Calculate the endpoints of the tick mark
        x1 = x_mid - (perp_ux * tick_length / 2)
        y1 = y_mid - (perp_uy * tick_length / 2)
        x2 = x_mid + (perp_ux * tick_length / 2)
        y2 = y_mid + (perp_uy * tick_length / 2)
        # Return the SVG line element for the tick mark
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"/>\n'

    # Draw tick marks on both line segments
    svg_cmd += draw_tick(A, B)
    svg_cmd += draw_tick(C, D)

    return svg_cmd


def draw_pair_of_lines(arg_pts, color, org_2_alpha_geo, point_name_2_loc):
    assert len(arg_pts) == 4
    A = point_name_2_loc[org_2_alpha_geo[arg_pts[0]].lower()]
    B = point_name_2_loc[org_2_alpha_geo[arg_pts[1]].lower()]
    C = point_name_2_loc[org_2_alpha_geo[arg_pts[2]].lower()]
    D = point_name_2_loc[org_2_alpha_geo[arg_pts[3]].lower()]
    # Draw the two line segments
    svg_cmd = (
        f'<line x1="{A[0]}" y1="{A[1]}" x2="{B[0]}" y2="{B[1]}" stroke="{color}" stroke-width="2"/>\n'
        f'<line x1="{C[0]}" y1="{C[1]}" x2="{D[0]}" y2="{D[1]}" stroke="{color}" stroke-width="2"/>\n')
    return A, B, C, D, svg_cmd


def draw_goal(goal_str, point_name_2_loc, org_2_alpha_geo, color):
    svg_cmd = ''
    if goal_str == '':
        return ''

    goal_cmd_w_args = goal_str.strip().split(' ')
    goal_cmd, arg_pts = goal_cmd_w_args[0], goal_cmd_w_args[1:]

    if goal_cmd.lower() == 'perp':
        svg_cmd += draw_perp(arg_pts, color, point_name_2_loc, org_2_alpha_geo)

    if goal_cmd.lower() == 'cong':
        svg_cmd += draw_congruent(arg_pts, color, point_name_2_loc, org_2_alpha_geo)

    return svg_cmd

