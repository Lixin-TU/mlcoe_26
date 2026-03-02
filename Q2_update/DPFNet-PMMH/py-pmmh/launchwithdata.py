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
#
#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import os
import sys
import scipy.weave as weave
from scipy.stats import norm
from scipy import special
from numpy import random, exp, sqrt, power, log, newaxis
# import pmcmc classes
from pmcmc.SIR import SIR
from pmcmc.PMMH import PMMH
from pmcmc.modele import HiddenStateModel
from pmcmc.customdist import CustomDist
# import stochastic volatility model, prior and proposal
from stochvolwithdata.stochvolmodel import *
from stochvolwithdata.stochvolprior import *
from stochvolwithdata.stochvolproposal import *

## If you want to change the prior,  check the file: stochvol/stochvolprior.py
## If you want to change the proposal, for example the metropolis hastings step, check the file: stochvol/stochvolproposal.py
## If you want to change the model, for example the number of observations, check the file: stochvol/stochvolmodel.py
## To make a new model, use the stochvol folder as a template

###################################################
## parameters to play with: init points, particles, iterations
initparam = {"mu": 0.0, "rho": 0.95, "sigma": 0.3}
npart = 1000
niter = 1000
###################################################

###################################################
## launch the algorithm
####

## initialize PMMH instance
pmmh = PMMH(initparam, model, prop, prior, step, N = niter, npart = npart, ESSresampling = True, resamplingmethod = "cpp")
## PMMH first iteration 
pmmh.first_iter()
## PMMH following iterations
pmmh.iter()
#

###################################################
## PMMH output in a results.R file, in the results folder
pmmh.output()
###################################################

###################################################
## a little bit of plottin'
from pmcmc.plot import Plot
pl = Plot(pmmh)
pl.plotaccept()
pl.plotparam()
pl.plotacf()
pl.plotcorr()
pl.latexify()
####
###################################################

###################################################
#### a little bit of profilin'...
#import cProfile
#cProfile.run("""pmmh.iter()""", "prof")
#import pstats
#p = pstats.Stats('prof')
#p.sort_stats("cumulative").print_stats(10)
#p.sort_stats("time").print_stats(10)
#p.sort_stats("calls").print_stats(10)
###################################################

