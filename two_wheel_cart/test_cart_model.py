import math
import unittest

import numpy as np

from cart_model import DifferentialDriveCart, SignalToWheels, render_target_camera


class CartModelTests(unittest.TestCase):
    def test_signal_is_amplified_without_crossing_sides(self):
        mapper = SignalToWheels(motor_gain=10.0, max_wheel_speed=2.0)
        command = mapper.command({"L": 0.04, "R": 0.1})
        self.assertAlmostEqual(command.left_speed, 0.4)
        self.assertAlmostEqual(command.right_speed, 1.0)

    def test_speed_is_clipped(self):
        mapper = SignalToWheels(motor_gain=100.0, max_wheel_speed=1.0)
        command = mapper.command({"L": 2.0, "R": -1.0})
        self.assertEqual(command.left_speed, 1.0)
        self.assertEqual(command.right_speed, -1.0)

    def test_equal_wheels_move_straight(self):
        cart = DifferentialDriveCart()
        cart.step(0.5, 0.5, 2.0)
        self.assertAlmostEqual(cart.x, 1.0)
        self.assertAlmostEqual(cart.y, 0.0)
        self.assertAlmostEqual(cart.heading, 0.0)

    def test_faster_right_wheel_turns_left(self):
        cart = DifferentialDriveCart()
        cart.step(0.1, 0.5, 0.2)
        self.assertGreater(cart.heading, 0.0)

    def test_left_target_is_rendered_in_left_image_half(self):
        cart = DifferentialDriveCart()
        image = render_target_camera(cart, (2.0, 1.0), width=100, height=40)
        x = int(np.unravel_index(image[..., 0].argmax(), image[..., 0].shape)[1])
        self.assertLess(x, 50)


if __name__ == "__main__":
    unittest.main()
