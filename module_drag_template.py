#!/usr/bin/env python
#LSF -q suncat2 -W 50:00 -n PROC -o KEY_mod_drag_METHOD.log -e KEY_mod_drag_METHOD.err

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
from DATABASE import *
from AidanDrag import AidanDrag
from collections2 import defaultdict
from bad_db_implementation import get

#######################################
#             PARAMETERS              #
#######################################

fname = sys.argv[0].split('/')[-1].split('.')[0]

pw=PWS
dw=DWS
kpts=KPTS
mol = 'KEY'
func = 'FUNC'
psppath = '/nfs/slac/g/suncatfs/aidank/scripts/gbrv/'
intermediate_images = IMGS
relax_num = RELAX_NUM
restart = 'RESTART'
#distances = 'DISTANCES'
#spring = 'SPRING'
#jump = 'JUMP'
prefix ='PREFIX'
#method='METHOD'
startDirectory='DIRECTORY'
#int_directory='INT_DIRECTORY'
zcheck = 'ZCHECK'
add_layers='ADD_LAYERS'
fixed_layers='FIXED_LAYERS'
method='METHOD'
datafile = 'DATAFILE'
metric='METRIC'
struc_dirs = 'STRUC_DIRS'
optimizer = 'OPTIMIZER'
verbose ='VERBOSE'
write = 'WRITE'
sampling = 'SAMPLING'
fit_type='FIT_TYPE'
degree = 'DEGREE'
weight = WEIGHT
add_end_points=ADD_END_POINTS
quick = 'QUICK'

if quick =='quick'.upper():
    quick = False
else:
    quick = eval(quick)
if fit_type=='fit_type'.upper():
    fit_type = 'sigmoid'
else:
    try:
        fit_type = int(fit_type)
    except:
        pass
if degree=='degree'.upper():
    degree = 3
else:
    try:
        degree = int(degree)
    except:
        pass
if optimizer == 'optimizer'.upper():
    optimizer = 'QuasiNewton'
if metric =='metric'.upper():
    metric ='energy'
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
if sampling =='sampling'.upper():
    sampling='standard'

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
images = DATABASE().get_system(mol,prelim=False,directory=startDirectory,struc_dirs=struc_dirs,intermediate_images=intermediate_images,zcheck=zcheck,add_layers=add_layers,fixed_layers=fixed_layers,method='')

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
#ad = AidanDrag(images[0],images[-1],intermediate_images,(np.min(ads_inds),np.max(ads_inds)),calc,path_type=method,relax_num=relax_num,metric=metric,path_sampling=sampling,traj='p0_%s_drag' % sampling, logfile='p0_%s_drag' % sampling,outdir='p0_%s_drag' % sampling,verbose=verbose,write =write,datafile=datafile,weight=weight,fit_type=fit_type,add_end_points=add_end_points)
ad = AidanDrag(images[0],images[-1],intermediate_images,(np.min(ads_inds),np.max(ads_inds)),calc,path_type=method,relax_num=relax_num,metric=metric,path_sampling=sampling,traj='seven_%s_drag' % sampling, logfile='seven_%s_drag' % sampling,outdir='seven_%s_drag' % sampling,verbose=verbose,write =write)#,datafile=datafile,degree=degree)
ad.run(quick=quick)

################################
#             RETURN           #
################################

#np.savez('Relaxation_results.npz',times=times,time=time,geosteps=geosteps,energies=energies,TS_energy=np.max(energies))
#write_trajectory('final_neb_results.traj',images)
