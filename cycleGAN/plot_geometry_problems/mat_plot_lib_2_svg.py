from sympy.geometry import Point


class Elements:
    def __init__(self, name='', label='', x1=0, x2=0, y1=0, y2=0, radius=0, center=Point(0, 0)):
        self.name = name
        self.center = Point(center.x, center.y)
        self.label = label
        self.x = self.x1 = x1
        self.y = self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.radius = radius


class SympyGeo2SVG:
    def __init__(self):
        self.all_points_x = []
        self.all_points_y = []
        self.svg_code = []
        self.min_x = None
        self.max_y = None
        self.scale = None
        self.width = 800
        self.height = 600
        self.padding = 50
        self.svg_code = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
                         '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="800" height="600">',
                         f'<rect width="{self.width}" height="{self.height}" fill="white" />']
        self.all_elements = []

    def _compute_scale_and_offset(self):
        self.min_x = min(self.all_points_x)
        self.max_y = max(self.all_points_y)
        # Scaling factors
        if (max(self.all_points_x) - self.min_x) != 0:
            scale_x = (self.width - 2 * self.padding) / (max(self.all_points_x) - self.min_x)
        else:
            scale_x = 10
        if (self.max_y - min(self.all_points_y)) != 0:
            scale_y = (self.height - 2 *self.padding) / (self.max_y - min(self.all_points_y))
        else:
            scale_y = 10
        self.scale = min(scale_x, scale_y)  # Use the smaller scale to fit all content


    def _sympy_geo_pt_to_svg_pt(self, point):
        x = self._transform_x(point.x)
        y = self._transform_y(point.y)

        return Point(x, y)

    # Transform function for coordinates
    def _transform_x(self, x):
        return self.padding + self.scale * (x - self.min_x)

    def _transform_y(self, y):
        return self.padding + self.scale * (self.max_y - y)

    # SVG elements with scaling
    def _svg_line(self, p1, p2, color='black', width=2):
        tp1 = self._sympy_geo_pt_to_svg_pt(p1)
        tp2 = self._sympy_geo_pt_to_svg_pt(p2)
        self.svg_code.append(
            f'<line x1="{tp1.x:.5f}" y1="{tp1.y:.5f}" x2="{tp2.x:.5f}" y2="{tp2.y:.5f}" stroke="{color}" '
            f'stroke-width="{width}" opacity="0.8" />')

    def _svg_circle(self, center, radius, stroke_color='blue', fill='none'):
        t_center = self._sympy_geo_pt_to_svg_pt(center)
        t_radius = self.scale * radius
        self.svg_code.append(f'<circle cx="{t_center.x:.5f}" cy="{t_center.y:.5f}" '
                             f'r="{t_radius:.5f}" stroke="{stroke_color}" '
                             f'stroke-width="2" fill="{fill}" />')

    def _svg_point(self, p, label, color='red', text_color='red'):
        tp = self._sympy_geo_pt_to_svg_pt(p)
        self.svg_code.append(f'<circle cx="{tp.x:.5f}" cy="{tp.y:.5f}" r="5" fill="{color}" />'
                             f'\n<text x="{(tp.x + 10):.5f}" y="{(tp.y + 10):.5f}" fill="{text_color}" '
                             f'font-size="10">{label}</text>')

    def add_line(self, A, B):
        self.all_points_x.append(A.x)
        self.all_points_x.append(B.x)
        self.all_points_y.append(A.y)
        self.all_points_y.append(B.y)
        self.all_elements.append(Elements(name='line', x1=A.x, x2=B.x, y1=A.y, y2=B.y))

    def add_point(self, label, point):
        self.all_points_y.append(point.y)
        self.all_points_y.append(point.y)
        self.all_elements.append(Elements(name='point', label=label, x1=point.x, y1=point.y))

    def add_circle(self, center, radius):
        self.all_points_x.append(center.x + radius)
        self.all_points_x.append(center.x - radius)
        self.all_points_y.append(center.y + radius)
        self.all_points_y.append(center.y - radius)
        self.all_elements.append(Elements(name='circle', center=center, radius=radius))

    def get_svg_code(self):
        self._compute_scale_and_offset()
        for element in self.all_elements:
            if element.name.lower() == 'point':
                self._svg_point(p=Point(element.x, element.y), label=element.label)
            elif element.name.lower() == 'line':
                self._svg_line(p1=Point(element.x1, element.y1), p2=Point(element.x2, element.y2))
            elif element.name.lower() == 'circle':
                self._svg_circle(element.center, element.radius)
            else:
                raise NotImplementedError(f'element of kind {element.name} cant be drawn')
        self.svg_code.append('</svg>')
        return '\n'.join(self.svg_code)
