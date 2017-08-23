#!/usr/bin/python3

##  Copyright (C) 2016 D S Pavan Kumar
##  dspavankumar [at] gmail [dot] com
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.


import sys
import numpy
from subprocess import Popen, PIPE

## Read output feature dimension
def read_output_feat_dim (exp):
    p = Popen (['am-info', exp+'/final.mdl'], stdout=PIPE)
    for line in p.stdout:
        if b'number of pdfs' in line:
            return int(line.split()[-1])

## Compute priors
def compute_priors (exp, ali_tr, ali_cv=None):
    dim = read_output_feat_dim (exp)    
    counts = numpy.zeros(dim)

    ## Prepare string
    ali_str = 'ark:gunzip -c ' + ali_tr+'/ali.*.gz '
    if ali_cv:
        ali_str += ali_cv+'/ali.*.gz '
    ali_str += '|'

    p = Popen(['ali-to-pdf', exp+'/final.mdl', ali_str, 'ark,t:-'], stdout=PIPE)

    ## Compute counts
    for line in p.stdout:
        line = line.split()
        for index in line[1:]:
            counts[int(index)] += 1

    ## Compute priors
    priors = counts / numpy.sum(counts)

    ## Floor zero values
    priors[priors==0] = 1e-5

    ## Write to file
    priors.tofile (exp+'/dnn.priors.csv', sep=',', format='%e')

if __name__ == '__main__':
    exp     = sys.argv[1]
    ali_tr  = sys.argv[2]
    ali_cv  = sys.argv[3] if len(sys.argv)==4 else None

    compute_priors (exp, ali_tr, ali_cv)
