#!/usr/bin/env python
# -*- encoding: ascii -*-
class Differentiator:
    """Numerical Differentiator
    using Tustin's rule or trapezoidal rule, refer to "Small Unmanned Aircraft"
    TODO: five-point numerical differentiation
        https://www3.nd.edu/~zxu2/acms40390F15/Lec-4.1.pdf
    """

    def __init__(self, Ts):
        self.x_delay_1 = 0.0  # delayed by one time step
        self.x_dot_delay_1 = 0.0

        # gains for differentiator
        tau = 0.05
        self.a1 = (2.0 * tau - Ts) / (2.0 * tau + Ts)
        self.a2 = 2.0 / (2.0 * tau + Ts)

    def update(self, x):
        x_dot = self.a1 * self.x_dot_delay_1 + self.a2 * (x - self.x_delay_1)
        self.x_delay_1 = x
        self.x_dot_delay_1 = x_dot
        return x_dot
