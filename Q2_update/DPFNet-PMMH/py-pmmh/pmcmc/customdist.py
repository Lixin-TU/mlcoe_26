###################################################
#    This file is part of py-pmmh.
#
#    py-pmmh is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    py-pmmh is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with py-pmmh.  If not, see <http://www.gnu.org/licenses/>.
###################################################

#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class to create a Custom Distribution
"""
from __future__ import division

class CustomDist:
    def __init__(self):
        self.parameters = dict()
    def setGenerator(self, function):
        """ custom dist random variate generator """
        self.Generator = function
    def setDensity(self, function):
        """ custom dist probability density function """
        self.Density = function

