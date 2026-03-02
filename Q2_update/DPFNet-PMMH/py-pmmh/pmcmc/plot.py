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
The Plot class is used after the PMCMC algorithm finishes, to
plot ACF, parameters plot, correlations between parameters, etc.
It also create a LaTeX file to show the EPS plots, and compiles it as long
as LaTeX is installed. Then it converts dvi to pdf using dvipdfm.
In the end the results are available in a separate folder, for conveniency.
"""
import subprocess
import rpy2.robjects as robjects
import os
import shutil

latexincludes = r"""
\documentclass[a4paper,11pt]{article}
\usepackage[french]{babel}
\usepackage[latin1]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{float}
\usepackage{fancyhdr}
\usepackage{geometry}
\usepackage{amsmath,amssymb,latexsym}
\usepackage{xspace}
"""
latexbase = r"""
\begin{document}

This document shows the results of a Particle Markov Chain Monte Carlo algorithm called ``Particle Marginal Metropolis Hastings'', described in the paper by Andrieu, Doucet and Holenstein. This algorithm was applied to a simple stochastic volatility model. The aim of the modelisation is to estimate the parameters\ldots

\begin{align*}
 y_t &\sim \mathcal{N}(0, e^{x_t})\\
 x_t &= \mu + \rho (x_{t-1} - \mu) + \sigma \varepsilon_t
\end{align*}

Here the algorithm was launched with \npart particles and \niter M-H iterations. The data on which it was applied is made of \ndata observations.

The M-H move was the following:
\begin{align*}
\mu^* &\sim \mathcal{N}(\mu, \stepmu^2) \\
\rho^* &\sim \mathcal{N}(\rho, \steprho^2) \\
\log{\sigma^*} &\sim \mathcal{N}(\log{\sigma}, \stepsigma^2) \\
\end{align*}

The results are presented in the following graphs:

\begin{figure}[H]
 \centering
 \includegraphics[width=0.6\textwidth]{parameters.eps}
\caption{\label{param} Parameter values for mu, rho and sigma, plotted against iteration indices.}
\end{figure}

\begin{figure}[H]
 \centering
 \includegraphics[width=0.6\textwidth]{acf.eps}
\caption{\label{acf} Autocorrelations of mu, rho and sigma series.}
\end{figure}

\begin{figure}[H]
 \centering
 \includegraphics[width=0.6\textwidth]{acceptrate.eps}
\caption{\label{accept} Acceptance rate of the Metropolis-Hastings algorithm.}
\end{figure}

\begin{figure}[H]
 \centering
 \includegraphics[width=0.6\textwidth]{corr.eps}
\caption{\label{corr} Correlations between pairs of variables.}
\end{figure}

\end{document}
"""

class Plot:
    def __init__(self, pmmhresult):
        self.r = robjects.r
        self.pmmh = pmmhresult
        self.savefolder = os.path.abspath(pmmhresult.savefolder)
        datafileend = "results.R"
        datafile = os.path.join(self.savefolder, datafileend)
        self.r.source("%s" % datafile)
        self.latexnewcommands = r"""
\newcommand{\niter}{%i\xspace}
\newcommand{\npart}{%i\xspace}
\newcommand{\ndata}{%i\xspace}
\newcommand{\stepmu}{%.5f\xspace}
\newcommand{\steprho}{%.5f\xspace}
\newcommand{\stepsigma}{%.5f\xspace}
        """ % (self.pmmh.niter, self.pmmh.npart, self.pmmh.model.length, self.pmmh.step["mu"], self.pmmh.step["rho"], self.pmmh.step["sigma"])
    def plotaccept(self):
        """ plot accept ratio """
        self.r("""
            txaccepts <- c()
            somme <- 0
            for (i in 1:(length(accepts))){
              somme <- somme + accepts[i]
              txaccepts <- c(txaccepts, somme / i)
            }
            plot(txaccepts, type = "l", col = "orange", main = "Acceptance rate of the MH algorithm")
            dev.copy2eps(file = "%s/acceptrate.eps")
        """ % self.savefolder)
    def plotparam(self):
        """ plot parameters """
        self.r("""
            par(mfrow = c(3, 1))
            plot(mus, type = "l", col = "red")
            plot(rhos, type = "l", col = "green")
            plot(sigmas, type = "l", col = "blue")
            dev.copy2eps(file = "%s/parameters.eps")
        """ % self.savefolder)
    def plotacf(self):
        """ plot autocorrelation functions """
        self.r("""
            par(mfrow = c(3, 1))
            acf(mus, lag.max = 1000, col = "red")
            acf(rhos, lag.max = 1000, col = "green")
            acf(sigmas, lag.max = 1000, col = "blue")
            dev.copy2eps(file = "%s/acf.eps")
        """ % self.savefolder)
    def plotcorr(self):
        """ plot pairwise correlations """
        self.r("""
            par(mfrow = c(3, 1))
            plot(x = mus, y = rhos, col = "red")
            plot(x = rhos, y = sigmas, col = "green")
            plot(x = sigmas, y = mus, col = "blue")
            dev.copy2eps(file = "%s/corr.eps")
        """ % self.savefolder)
    def latexify(self):
        """ create the latex file, compiles it and converts the dvi file to pdf using dvipdfm """
        latexcontent = latexincludes + self.latexnewcommands + latexbase
        latexfile = open(os.path.join(self.savefolder, "figures.tex"), "w")
        latexfile.write(latexcontent)
        latexfile.close()
        os.chdir(self.savefolder)
        try:
            subprocess.call(["latex", "figures.tex"])
        except:
            print "latex compile didnt work (is latex installed ?)"
        try:
            subprocess.call(["dvipdfm", "-o", "figures.pdf", "figures.dvi"])
        except:
            print "dvi to pdf conversion failed (is dvipdfm installed ?)"




