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
import scipy.weave as weave
from numpy import random, array, sqrt, empty, double, log, exp, sum, column_stack, mean, arange, ones, power, cumsum, zeros
#from numpy import array


class SIR:
    def __init__(self, HMMmodel, parameters, npart = 0, init = False, ESSresampling = False, resamplingmethod = "python"):
        self.nparticles = npart
        self.model = HMMmodel
        self.states = zeros((self.nparticles, self.model.length))
        self.weights= zeros((self.nparticles, self.model.length))
        self.incr_weights= zeros((self.nparticles, self.model.length))
        self.model.setParameters(parameters)
        self.ESSresampling = ESSresampling
        self.ESSs = zeros(self.model.length-1)
        if resamplingmethod == "cpp":
            self.resample = self.IndResample
        else:
            self.resample = self.IndResample_slow
        if init:
            self.first_step()
            self.following_steps()
    def __del__(self):
    	del self.states
        del self.weights
        del self.incr_weights
        del self.ESSs

    def setNparticles(self, n):
        self.nparticles = n
    def IndResample_slow(self, w):
        """ Indexed of resampled particles
        (deterministic scheme) """
        u = random.uniform(size = 1, low = 0, high = 1)
        cnw = (self.nparticles / sum(w)) * cumsum(w)
        j = 0
        Ind = empty(self.nparticles, dtype="int")
        for k in xrange(self.nparticles):
                while cnw[j] < u:
                        j = j+1
                Ind[k] = j
                u = u + 1.
        return Ind
    def IndResample(self, w):
        """ Indexed of resampled particles
        (deterministic scheme) """
        code = \
        """ int j = 0;
            double csw = 0;
            for(int k = 0; k < H; k++)
            {
                while(csw < u && j < H - 1)
                {
                    j++;
                    csw += w(j);
                }
                Ind(k) = j;
                u = u + 1.;
            }

        """ 
        u = float(random.uniform(size = 1, low = 0, high = 1)[0])
        H = self.nparticles
        w = double(w * H / sum(w))
        Ind = empty(H, dtype='int')
        weave.inline(code,['u','H','w','Ind'], type_converters=weave.converters.blitz)
        return Ind

    def first_step(self):
        states = random.normal(size = self.nparticles, loc = 0, scale = sqrt(5))
        y = self.model.y[0]
        self.incr_weights[:,0] = self.model.vectMeasureDensity(y, states)
        self.weights[:,0] = self.incr_weights[:,0]
    def following_steps(self):
        for time in xrange(1, self.model.length):
            #print time
            y = self.model.y[time]
            ESS = self.ESS(self.weights[:, time - 1])
            self.ESSs[time - 1] = ESS
            if ESS < 0.5 * self.nparticles or not(self.ESSresampling):
                #print "resampling"
                indices = self.resample(self.weights[:,time -1])
                weights = ones(self.nparticles)
            else:
                #print "not resampling"
                indices = arange(self.nparticles)
                weights = self.weights[:, time - 1]
            states = self.states[:,time -1][indices]
            #transition:
            states = self.model.vectTransitionGenerator(states, time = 0)
            self.states[:,time] = states
            #weighting
            incr_weights = self.model.vectMeasureDensity(y, states)
            self.incr_weights[:,time] = incr_weights
            self.weights[:,time]  = incr_weights * weights
    def llikelihood(self):
        means = mean(self.incr_weights, axis = 0)
        return sum(log(means))
    def ESS(self, weights):
        norm_weights = weights / sum(weights)
        sqweights = power(norm_weights, 2)
        return 1 / sum(sqweights)



