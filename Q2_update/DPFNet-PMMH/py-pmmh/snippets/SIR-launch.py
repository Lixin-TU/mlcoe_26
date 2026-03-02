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

from __future__ import division
import os
import sys
import scipy.weave as weave
from scipy.stats import norm
from scipy import special
from numpy import random, exp, sqrt, power, log, average
basepath = os.path.dirname(os.path.abspath("."))
sys.path.append(basepath)
from pmcmc.SIR import SIR
from pmcmc.modele import HiddenStateModel
from stochvol.stochvolmodel import *

npart = 100

###################################################
## launch the algorithm
####

s = SIR(model, param, npart, init = True, ESSresampling = True, resamplingmethod = "python")

#### a little bit of profiling
#import cProfile
#cProfile.run("""s = SIR(model, param, npart, init = True, ESSresampling = True, resamplingmethod = "cpp")""", "prof")
#import pstats
#p = pstats.Stats('prof')
#p.sort_stats("cumulative").print_stats(10)
#p.sort_stats("time").print_stats(10)
#p.sort_stats("calls").print_stats(10)
#
## graph results with rpy2
#
#import rpy2.robjects as robjects
#r = robjects.r
#mx = robjects.FloatVector(model.x)
#ess = robjects.FloatVector(s.ESSs)
#xmeans = robjects.FloatVector(average(s.states, weights=s.weights, axis = 0))
#r("par(mfrow = c(2,1))")
#r.plot(mx, ylab = "mx", type = "l", ylim = robjects.FloatVector([-5, 5]))
#r.lines(xmeans, col = "green")
#r.plot(ess, ylab = "ess", type = "l")
#raw_input("appuyez sur une touche pour fermer la fenetre")

