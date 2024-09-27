from random_generation.draw_svg.get_svg import draw_goal
import unittest


class TestGoalDrawing(unittest.TestCase):

    def setUp(self):
        # This is the correct method to set up shared state for tests.
        self.point_name_2_loc = {
            'A': (100, 120),
            'B': (200, 120),
            'C': (100, 10),
            'D': (100, 110)
        }
        self.forward_point_name_map = {
            'A': 'A',
            'B': 'B',
            'C': 'C',
            'D': 'D'
        }
        self.color = 'red'

    def test_draw_perp(self):
        # Sample input data
        goal_str = 'perp A B C D'

        # Call the function
        svg_output = draw_goal(goal_str, self.point_name_2_loc, self.forward_point_name_map, self.color)

        # print(svg_output)

        # Expected SVG output
        expected_svg = (
            '<line x1="100" y1="120" x2="200" y2="120" stroke="red" stroke-width="2"/>\n'
            '<line x1="100" y1="10" x2="100" y2="110" stroke="red" stroke-width="2"/>\n'
            '<path d="M 100.0 120.0 L 110.0 120.0 L 110.0 110.0 L 100.0 110.0 Z" fill="none" stroke="red" '
            'stroke-width="2"/>\n'
        )

        # Assert that the output matches the expected SVG
        self.assertEqual(svg_output, expected_svg)

    def test_draw_cong(self):
        # Sample input data
        goal_str = 'cong A B C D'

        # Call the function
        svg_output = draw_goal(goal_str, self.point_name_2_loc, self.forward_point_name_map, self.color)

        # print(f'\n{svg_output}')

        # Expected SVG output
        expected_svg = ('<line x1="100" y1="120" x2="200" y2="120" stroke="red" stroke-width="2"/>\n'
                        '<line x1="100" y1="10" x2="100" y2="110" stroke="red" stroke-width="2"/>\n'
                        '<line x1="150.0" y1="115.0" x2="150.0" y2="125.0" stroke="red" stroke-width="2"/>\n'
                        '<line x1="105.0" y1="60.0" x2="95.0" y2="60.0" stroke="red" stroke-width="2"/>\n')


        # Assert that the output matches the expected SVG
        self.assertEqual(svg_output, expected_svg)

if __name__ == '__main__':
    unittest.main()
