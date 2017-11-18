#!/usr/bin/env python
#LSF -q suncat2 -W 50:00 -n 12 -o Ag_NH2-H_BS_fix_idpp.log -e Ag_NH2-H_BS_fix_idpp.err

import sys
import os
import os.path
import numpy as np
import math
import subprocess
from ase import Atoms
from ase import io
from ase.io.trajectory import write_trajectory
from ase.constraints import FixBondLength,FixedLine, FixedPlane,FixAtoms
from ase.neb2 import NEB
from ase.structure import molecule
from ase.lattice.surface import *
#from ase.structure import bulk
from ase.optimize import QuasiNewton,BFGS,FIRE
from ase.data import ground_state_magnetic_moments, atomic_numbers
#sys.path.insert(0,'/nfs/slac/g/suncatfs/vossj/espressopre3')
from espresso import espresso
from espresso.multiespresso import multiespresso
sys.path.insert(0,'/u/if/aidank/nfs/scripts')
from myparser import *
sys.path.insert(0,'/u/if/aidank/nfs/NEB_benchmarking')
#from relax_checker import *
from NEBFragmentDatabase import *
from AidanDrag import AidanDrag
from collections2 import defaultdict
from bad_db_implementation import get

#######################################
#             PARAMETERS              #
#######################################

fname = sys.argv[0].split('/')[-1].split('.')[0]

pw=400
dw=3200
kpts=(2, 4, 1)
mol = 'Ag_NH2-H'
func = 'pbe'
psppath = '/nfs/slac/g/suncatfs/aidank/scripts/gbrv/'
intermediate_images = 10
relax_num = 10
restart = 'RESTART'
#distances = 'DISTANCES'
#spring = 'SPRING'
#jump = 'JUMP'
prefix ='PREFIX'
#method='idpp'
startDirectory='/nfs/slac/g/suncatfs/aidank/NEB_benchmarking/NEBRun'
#int_directory='INT_/nfs/slac/g/suncatfs/aidank/NEB_benchmarking/NEBRun'
zcheck = 'True'
add_layers='0'
fixed_layers='2'
method='idpp'
struc_dirs = '["newInitialStructures", "newFinalStructures"]'
optimizer = 'OPTIMIZER'
verbose ='True'
write = 'True'

if optimizer == 'optimizer'.upper():
    optimizer = 'QuasiNewton'
#if spring == 'spring'.upper():
#    spring= 0.15
#else:
#    spring = float(spring)
#if jump=='jump'.upper():
#    jump =False
if restart == 'restart'.upper():
    restart = False
else:
    restart = eval(restart)
#if distances == 'distances'.upper():
#    distances = []
#else:
#    distances=eval(distances)
if prefix =='prefix'.upper():
    prefix=''
if startDirectory =='directory'.upper():
    startDirectory = '/nfs/slac/g/suncatfs/aidank/NEB_benchmarking/newFragmentTest'
if struc_dirs == 'struc_dirs'.upper():
    struc_dirs=['InitialStructures','FinalStructures']
else:
    struc_dirs=eval(struc_dirs)
if zcheck=='zcheck'.upper():
    zcheck=False
if add_layers=='add_layers'.upper():
    add_layers=0
else:
    add_layers=int(add_layers)
if fixed_layers=='fixed_layers'.upper():
    fixed_layers=2
else:
    fixed_layers = int(fixed_layers)
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

if prefix:
    prefix+='_'
#if len(distances)>0:
#    intermediate_images = len(distances)
#if jump:
#    img_num=-10
#else:
#    img_num=-1

energies=[]
output = {'avoidio':True,'removewf':True,'wf_collect':False,'removesave':True}
convergence = {'energy':13.6*1.e-6, 'mixing':0.1, 'nmix':15, 'mix':4, 'maxsteps':350,'diag':'david'}

#################################
#              ATOMS            #
#################################

#Set up an initial run
images = NEBFragmentDatabase().get_system(mol,prelim=False,directory=startDirectory,struc_dirs=struc_dirs,intermediate_images=intermediate_images,zcheck=zcheck,add_layers=add_layers,fixed_layers=fixed_layers,method=method)

################################
#           CALCULATOR         #
################################

#outdir = mol+'_drag'
calc = espresso(pw=pw,dw=dw,xc=func,kpts=kpts,nbands=-30,spinpol=False,smearing='fd',sigma=0.1,dipole={'status':True},output=output,convergence=convergence,psppath=psppath)


#################################
#         OPTIMIZE/NEB          #
#################################

#Fix rxn coord eventually
ads_inds = [j for j,a in enumerate(images[1]) if a.tag==0]
ad = AidanDrag(images[0],images[-1],intermediate_images,(np.min(ads_inds),np.max(ads_inds)),calc,path_type=method,relax_num=relax_num,metric='force',path_sampling='bisection',traj='%s_bisect' % method, logfile='%s_bisect' % method,outdir='%s_bisect' % method, verbose=verbose,write =write)
ad.run()

################################
#             RETURN           #
################################

#np.savez('Relaxation_results.npz',times=times,time=time,geosteps=geosteps,energies=energies,TS_energy=np.max(energies))
#write_trajectory('final_neb_results.traj',images)
