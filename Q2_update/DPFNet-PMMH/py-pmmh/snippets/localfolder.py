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

import pat  # touch pat.py in the cwd and try to import the empty file under Linux
import os

################## os.getcwd()##############
def get_path():
    PAT=str(pat).split()[3][1:-9] # PATH extracted..
    try:
        os.remove(PAT + 'pat.pyc')# get_rid...
    except OSError:
        PAT=PAT +'/'
        try:
            os.remove(PAT + 'pat.pyc')# Fix for mutiple calls..  
        except:
            pass
    PATpath = os.path.abspath(PAT)
    return os.path.dirname(PATpath)
###############################

