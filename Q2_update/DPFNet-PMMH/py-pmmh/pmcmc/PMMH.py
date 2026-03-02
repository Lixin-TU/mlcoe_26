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
import numpy
import os
from numpy import random, log
from SIR import SIR
from snippets.localfolder import get_path

class PMMH:
    def __init__(self, parameters, model, proposaldist, priordist, step, N = 0, npart = 0, init = False, ESSresampling = True, resamplingmethod = "cpp"):
        self.niter = N
        self.npart = npart
        self.currentparameters = parameters 
        self.parameterslist = [parameters]
        self.llikelihoods = []
        self.currentllikelihood = 0
        self.accepts = []
        self.model = model
        self.step = step 
        self.proposal = proposaldist
        self.prior = priordist 
        self.ESSresampling = ESSresampling
        self.resamplingmethod = resamplingmethod
        self.savefolder = ""
        if init:
            self.first_iter()
            self.iter()
            self.output()
    def first_iter(self):
        print "PMMH: initialization"
        s = SIR(self.model, self.currentparameters, npart = self.npart, init = True, ESSresampling = self.ESSresampling, resamplingmethod = self.resamplingmethod)
        self.currentllikelihood = s.llikelihood()
        s.__del__()
        self.llikelihoods.append(self.currentllikelihood)
    def iter(self):
        for it in xrange(1, self.niter):
            print "PMMH: iter %i / %i, with %i particles, on %i observations" % (it, self.niter, self.npart, self.model.length)
            parameters_star = self.proposal.Generator(self.step, self.currentparameters)
            if parameters_star["rho"] > 1 or parameters_star["rho"] < -1:
                continue
            SIR_star = SIR(self.model, parameters_star, npart = self.npart, init = True, ESSresampling = self.ESSresampling, resamplingmethod = self.resamplingmethod)
            llikelihood_star = SIR_star.llikelihood()
            SIR_star.__del__()
            numerator =  \
            llikelihood_star + log(self.prior.Density(parameters_star)) + log(self.proposal.Density(self.currentparameters))
            denominator = \
            self.currentllikelihood + log(self.prior.Density(self.currentparameters)) + log(self.proposal.Density(parameters_star))
            logMHratio = numerator - denominator
            if log(random.uniform(size = 1, low = 0, high = 1)[0]) < logMHratio:
                #print "move accepted"
                self.currentparameters = parameters_star
                self.currentllikelihood = llikelihood_star
                self.accepts.append(1)
            else:
                #print "move rejected"
                self.accepts.append(0)
            self.llikelihoods.append(self.currentllikelihood)
            self.parameterslist.append(self.currentparameters)
        print "final parameter values:", self.currentparameters
    def createfolder(self, resultsfolder = ""):
        if resultsfolder == "":
            resultsfolder = get_path()
            resultsfolder = os.path.join(resultsfolder, "results")
        self.basesavedirectory = os.path.abspath(resultsfolder)
        self.details="i%i-p%i-d%i-step%.4f_%.4f_%.4f"%(self.niter, self.npart, self.model.length, self.step["mu"], self.step["rho"], self.step["sigma"])
        self.subfoldername = "results-%s" % self.details
        self.savefolder = os.path.join(self.basesavedirectory, self.subfoldername)
        if not(os.path.isdir(self.savefolder)):
            os.mkdir(self.savefolder)
            print "folder created: %s" % self.savefolder
        else:
            compteur = 2
            self.savefoldertest = self.savefolder
            while os.path.isdir(self.savefoldertest):
                self.savefoldertest = self.savefolder + "(%i)" % compteur
                compteur += 1
            self.savefolder = self.savefoldertest
            os.mkdir(self.savefolder)
            print "folder created: %s" % self.savefolder
    def output(self, resultsfolder = ""):
        self.createfolder(resultsfolder)
        mus = [str(param["mu"]) for param in self.parameterslist]
        rhos = [str(param["rho"]) for param in self.parameterslist]
        sigmas = [str(param["sigma"]) for param in self.parameterslist]
        accepts= [str(accept) for accept in self.accepts]
        tempfilename = os.path.join(self.savefolder, "results.R")
        tempfile = open(tempfilename, "w")
        tempfile.write("mus <- c(" + ",".join(mus) + ")\n")
        tempfile.write("rhos <- c(" + ",".join(rhos) + ")\n")
        tempfile.write("sigmas <- c(" + ",".join(sigmas) + ")\n")
        tempfile.write("accepts <- c(" + ",".join(accepts) + ")\n")
        tempfile.close()
        




