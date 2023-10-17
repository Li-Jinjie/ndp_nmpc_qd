#!/usr/bin/env python
# -*- encoding: ascii -*-
"""
Author: LI Jinjie
File: alpha_filter.py
Date: 2023/4/18 10:57 AM
Description:
"""


class AlphaFilter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        self.y = self.alpha * self.y + (1 - self.alpha) * u
        return self.y
