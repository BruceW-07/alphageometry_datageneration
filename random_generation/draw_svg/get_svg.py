from typing import Any, Optional, Union
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

    # Draw points
    for point in points:
        svg_code = draw_point_svg(point.num, point.name, lines, circles, transform, color='red')
        svg_elements.append(svg_code)

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
    p: Point,
    name: str,
    lines: list[Line],
    circles: list[Circle],
    transform,
    color: str = 'black',
    size: float = 5,
) -> str:
    """Return SVG code for drawing a point."""
    # Transform the point coordinates
    tp = transform(p)

    # Generate the circle element
    svg_code = f'<circle cx="{tp.x}" cy="{tp.y}" r="{size}" fill="{color}" />\n'

    # Determine position for the label
    x_label, y_label = tp.x + 10, tp.y - 10  # Simple offset for label

    # Add the text element
    svg_code += f'<text x="{x_label}" y="{y_label}" fill="{color}" font-size="{size * 2}">{name.upper()}</text>\n'

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
    svg_code = f'<circle cx="{center.x}" cy="{center.y}" r="{radius}" stroke="{color}" stroke-width="{lw}" fill="none" />\n'
    return svg_code
