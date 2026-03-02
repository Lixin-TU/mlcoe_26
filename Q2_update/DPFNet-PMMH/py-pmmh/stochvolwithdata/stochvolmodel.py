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

from pmcmc.modele import HiddenStateModel
from numpy import random, power, sqrt, exp
from scipy.stats import norm
###################################################

###################################################
## Set the Hidden Markov Model as a stochastic volatility model
####
def firstStateGenerator(parameters):
    return(random.normal(size = 1, loc = parameters["mu"], scale = parameters["sigma"] / sqrt(1 - power(parameters["rho"], 2)))[0])
def observationGenerator(state):
    return(exp(state/2) * random.normal(size = 1, loc = 0, scale = 1)[0])
def vectorTransition(states, time, parameters):
    bruit = random.normal(size = len(states), loc = 0, scale = parameters["sigma"])
    return parameters["mu"] + parameters["rho"] * (states - parameters["mu"]) + bruit
def vectorMeasure(observation, states):
    return norm.pdf(observation, loc = 0, scale = exp(states / 2))
# parameters
# HMM model
model = HiddenStateModel()
#param = {"mu": 1, "rho": 0.9, "sigma": 0.4}
#model.setParameters(param)
model.setFirstStateGenerator(firstStateGenerator)
model.setVectTransitionGenerator(vectorTransition)
model.setMeasureGenerator(observationGenerator)
model.setVectMeasureDensity(vectorMeasure)
model.importData("/home/pierre/py-pmmh-data/poundd.dat")
#ndata = 100
#model.fakeData(ndata)


####
###################################################


