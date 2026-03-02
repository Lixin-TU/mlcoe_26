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
The HiddenStateModel class allows to specify a Hidden State model.
The Ys then denote the observations,
the Xs denote the hidden states.
Transition equation and measure equation are specified.
"""
from __future__ import division
import numpy

class HiddenStateModel:
    """ Hidden State Model Class"""
    def __init__(self):
        """ init with no argument """
    	self.x = []
        self.y = []
        self.length = 0
    def setParameters(self, p):
        """ set model parameters """
        self.parameters = p
    def setFirstStateGenerator(self, function):
        def appliedF():
            return(function(self.parameters))
        self.firstStateGenerator = appliedF 
    def setTransitionDensity(self, function):
        """ transition density, pointwise """
        self.transitionDensity = function
    def setVectTransitionGenerator(self, function):
        """ transition generator, for a whole vector (much quicker than a loop) """
        def appliedF(states, time):
            return function(states, time, self.parameters)
        self.vectTransitionGenerator = appliedF
#    def setMeasureDensity(self, function):
#        """ measure density, pointwise """
#        self.measureDensity = function
    def setVectMeasureDensity(self, function):
        """ measure density, for a whole vector (much quicker than a loop)"""
        self.vectMeasureDensity = function
    def setMeasureGenerator(self, function):
        """ measure generator, used to generate fake data points """
        self.measureGenerator = function
    def fakeDataX(self, length):
        """ simulate hidden states X """
        #self.length = length
        self.x.append(self.firstStateGenerator())
        for k in range(1, length):
            self.x.append(self.vectTransitionGenerator(numpy.array([self.x[-1]]), k)[0])
    def fakeDataY(self):
        """ simulate observations Y, given already generated hidden states """
        self.y = [self.measureGenerator(state) for state in self.x]
    def fakeData(self, length):
        """ simulate hidden states and observations """
    	self.fakeDataX(length)
    	self.fakeDataY()
        self.length = length
    def importData(self, filename):
        """ import data from a specified file """
        f = open(filename, "r")
        importeddata = f.read()
        f.close()
        ly = importeddata.split("\n")
        ly = [y for y in ly if len(y) > 0]
        ly = [float(y) for y in ly]
        self.y = ly
        self.length = len(ly)
        print "%i observations imported from file %s" % (self.length, filename)
    def plotData(self):
        import rpy2.robjects as robjects
        ry = robjects.FloatVector(self.y)
        robjects.r.plot(ry, type = "l", lwd = 1, ylab = "y")



