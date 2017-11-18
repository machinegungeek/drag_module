#!/usr/bin/env python
#LSF -q suncat2 -W 50:00 -n 12 -o Ag_NH2-H_BS_fix_idpp.log -e Ag_NH2-H_BS_fix_idpp.err

import sys
import os
import os.path
import numpy as np
import math
from ase.io import PickleTrajectory
from ase.io.trajectory import write_trajectory
from ase.constraints import FixBondLength,FixedLine, FixedPlane,FixAtoms
from ase.neb2 import NEB
from ase.optimize import QuasiNewton,BFGS,FIRE
from espresso import espresso
sys.path.insert(0,'/u/if/aidank/nfs/NEB_benchmarking')
from AidanDrag import AidanDrag

#######################################
#             PARAMETERS              #
#######################################

pw=400
dw=3200
kpts=(2, 4, 1)
mol = 'Ag_NH2-H'
func = 'pbe'
psppath = '/nfs/slac/g/suncatfs/aidank/scripts/gbrv/'
intermediate_images = 10
relax_num = 10
optimizer = 'OPTIMIZER'
verbose ='True'
write = 'True'

if optimizer == 'optimizer'.upper():
    optimizer = 'QuasiNewton'
if method =='method'.upper():
    method=''
if verbose =='verbose'.upper():
    verbose = False
else:
    verbose = eval(verbose)
if write =='write'.upper():
    write=False
else:
    write = eval(write)

output = {'avoidio':True,'removewf':True,'wf_collect':False,'removesave':True}
convergence = {'energy':13.6*1.e-6, 'mixing':0.1, 'nmix':15, 'mix':4, 'maxsteps':350,'diag':'david'}

#################################
#              ATOMS            #
#################################

#Set up an initial and final states
initial = PickleTrajectory('Ag_NH2-H_IS.traj')[-1]
final = PickleTrajectory('Ag_NH2_FS.traj')[-1]
images = [initial,final]

################################
#           CALCULATOR         #
################################

calc = espresso(pw=pw,dw=dw,xc=func,kpts=kpts,nbands=-30,spinpol=False,smearing='fd',sigma=0.1,dipole={'status':True},output=output,convergence=convergence,psppath=psppath)


#################################
#         OPTIMIZE/NEB          #
#################################

#Fix rxn coord eventually
ads_inds = [j for j,a in enumerate(images[1]) if a.tag==0]
ad = AidanDrag(images[0],images[-1],intermediate_images,(np.min(ads_inds),np.max(ads_inds)),calc,path_type=method,relax_num=relax_num,metric='force',path_sampling='bisection',traj='%s_bisect' % method, logfile='%s_bisect' % method,outdir='%s_bisect' % method, verbose=verbose,write =write)
ad.run()

