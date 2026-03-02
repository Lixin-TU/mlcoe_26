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

step = {"mu": 0.00001, "rho": 0.00001, "sigma": 0.00001}

def dproposal(parameters):
    return 1 / parameters["sigma"]
def rproposal(step, parameters):
    newparameters = dict()
    newparameters["mu"] = random.normal(size = 1, loc = parameters["mu"], scale = step["mu"])[0]
    newparameters["rho"] = random.normal(size = 1, loc = parameters["rho"], scale = step["rho"])[0]
    newparameters["sigma"] = exp(random.normal(size = 1, loc = log(parameters["sigma"]), scale = step["sigma"])[0])
    return newparameters

prop = CustomDist()
prop.setGenerator(rproposal)
prop.setDensity(dproposal)

