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
from scipy import special
from scipy.stats import norm
from numpy import random, exp, sqrt, power, log, newaxis
from pmcmc.customdist import CustomDist

def invgammapdf(x, shape, intensity):
    return power(intensity, shape) / special.gamma(shape) * power(x, -shape-1) * exp(-intensity/x)
def dprior(parameters):
    return (norm.pdf(parameters["mu"], loc = 0, scale = 2) *
            invgammapdf(power(parameters["sigma"], 2), 3, 1))

prior = CustomDist()
prior.setDensity(dprior)

