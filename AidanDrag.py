import numpy as np
import cPickle as pickle
import os
import sys
import copy
from ase import Atoms,io
from ase.io.trajectory import write_trajectory
from ase.constraints import FixBondLength, FixedPlane
from ase.optimize import QuasiNewton,BFGS
from ase.calculators.calculator import Calculator
#Change for actual version
from ase.neb import NEB

class AidanDrag:
    """Class for running various forms of the drag method for finding transition states.
    All drag methods require the input of an initial and final state, the number of images to be created, 
    a reaction coordinate defining the relaxation constraints, and a calculator object for optimizations.
    The output is a numpy file, a pickle file, and a set of traj files, corresponding to the intermediate images. 
    The npz file contains lists of relaxed and unrelaxed energies, number of relaxations needed, and projected forces.
    These are in bisection (not reaction coordinate) order.
    The pckl file just contains a list of the relaxed energies, in reaction coordinate order.
    The traj files are also in reaction coordinate order.
    initial_state and final_state: ASE atoms objects. Should already be relaxed. Constraints will be passed on to other images.
    num_imgs: an integer. Number of intermediate images to create along the reaction path.
    rxn_coord: a tuple of atom indicies (a1,a2) for a fixed bond length calc or a vector to relax perp. to.
    calculator: An ASE calculator object to use for force calculations. Will be copied to all images.
    path_type: 'linear' or 'idpp'. Determines the creation of the initial path. Either linear or IDPP interpolation.
    relax_num: An integer. If not 0, determines how many images are greedily relaxed. If 0, all images are relaxed.
    metric: 'force' or 'energy'. Metric used to select images to greedily relax.
    'force' chooses those with the lowest tangental forces. 'energy', those with the highest potential energies.
    path_sampling: 'standard' or 'bisection'. How the path images are distributed. 
    'standard' is just an even sampling. 'bisection' bisects in the direction of lowest force, sampling the TS region more.
    optimizer: An ASE optimizer object. The optimizer used for the image relaxations If None, then QuasiNewton is used.
    logfile: String. Name for the optimizer logfiles (for relaxing drag images).
    traj: String. Name for the trajectory files of the relaxed drag images.
    outdir: String. Directory prefix for the path (if 'bisection' is used) and drag image calculations"""
    def __init__(self,initial_state, final_state, num_imgs,rxn_coord, calculator,path_type='linear',relax_num=-1,metric='energy',path_sampling = 'standard',optimizer = None,logfile='',traj='',outdir='',verbose=False,write=False,**kwargs):
        self.IS = initial_state
        self.FS = final_state
        self.nimgs = num_imgs
        self.rxn_coord = rxn_coord
        self.calc = calculator
        self.path_type = path_type
        #self.datafile=datafile
        self.verbose=verbose
        self.write = write
        self.kwargs = kwargs
        if relax_num<=0:
            self.relax_num = self.nimgs+relax_num
        else:
            self.relax_num = relax_num
        if not optimizer:
            self.optimizer = QuasiNewton
        else:
            self.optimizer = optimizer
        self.metric = metric
        self.sampling = path_sampling
        if not logfile:
            self.logfile='optDrag'
        else:
            self.logfile = logfile
        if not traj:
            self.traj='Drag'
        else:
            self.traj=traj
        if not outdir:
            self.outdir='Drag'
        else:
            self.outdir=outdir

    #The main method. Creates the path and relaxes the appropriate images. 
    #Returns the used trajectory files and a pckl file with a list of energies.
    def run(self,quick=False):
        #First create the initial path
        if self.sampling == 'bisection':
            self.path,self.energy_list,self.force_list= self.create_bisection_path()
        elif self.sampling =='standard':
            if self.write:
                self.inds = range(self.nimgs)
            if self.path_type =='linear' or self.path_type =='Linear':
                self.path = self.linear_interpolation(self.IS,self.FS,self.nimgs)
            elif self.path_type =='IDPP' or self.path_type =='idpp':
                #self.path = self.linear_interpolation(self.IS,self.FS,self.nimgs)
                #self.path = [self.IS] + [self.IS.copy()]*self.nimgs + [self.FS]
                self.path = [self.IS]
                for i in range(self.nimgs):
                    self.path+=[self.IS.copy()]
                self.path+=[self.FS]
                neb_=NEB(self.path,k=0.15,climb=True)
                neb_.interpolate(method='idpp')
                #self.mod_idpp_interpolate(self.path,traj=None,log=None,mic=False)
            #elif self.path_type=='fit':
            #    self.energy_inds=[]
                #assert self.datafile is not None
            #    from fitted_path import activity_path,direct_path, loocv_path
            #    assert 'datafile' in self.kwargs
                #if 'weight' not in self.kwargs:
                    #self.kwargs['weight']=0.
            #    if 'fit_type' not in self.kwargs and 'degree' not in self.kwargs:
            #        self.kwargs['fit_type']=3
                #self.path = loocv_path(self.IS,self.FS,self.nimgs,self.datafile,weight=0.,degree=1.,rel_rc=True)
                #self.path = direct_path(self.IS,self.FS,self.nimgs,self.kwargs.pop('datafile'),**self.kwargs)
            #    self.path = activity_path(self.IS,self.FS,self.kwargs.pop('datafile'),**self.kwargs)
            else: 
                raise ValueError('%s is an invalid input for the path_type. Please use either "linear" or "IDPP".' % self.path_type)             
            for img in self.path:
                self.attach_calculator(img)
        else:
            raise ValueError('%s is an invalid input for path_sampling. Please use "bisection" or "standard".' % self.path_sampling)   
        #Write the path for posterity. Verbose option maybe?
        write_trajectory(self.traj+'Path.traj',self.path)
        
        #Next, select the images to be relaxed
        assert self.relax_num<=self.nimgs
        if self.relax_num == self.nimgs:
            if self.verbose:
                self.select_images()
            self.relax_imgs=self.path[1:-1]
        elif self.relax_num >0:
            self.relax_imgs=self.select_images()
        else:
            if self.verbose:
                self.select_images()
            else:
                pass
        
        #Verbose..... inits, og_forces, and tangent should be temporary
        if self.verbose:
            pickle.dump({'Forces':self.force_list,'Energies':self.energy_list,'Inits':self.inits,'OG_Forces':self.og_forces,'Indices':self.inds},open(self.logfile+'_extra.pckl','wb'))
        if not quick:
            #Now relax the images...
            if hasattr(self,'relax_imgs'):
                energies=self.relax_images()
                #Output?
                TS_ind = np.argmax(energies)
                TS_energy = energies[TS_ind]
                #Add in option here
                pickle.dump(energies,open(self.logfile+'Results.pckl','wb'))
            if self.write:
                np.savez('Relaxation_results.npz',times=[0]*self.nimgs,inds=self.inds,geosteps=np.array(self.geosteps)[self.inds],energies = np.array(energies)[self.inds], forces=np.array(self.force_list)[self.inds],old_energies=np.array(self.energy_list)[self.inds])    

    #Simple linear interpolation. num_imgs=intermediate images
    def linear_interpolation(self,initial_state, final_state,num_imgs):
        path = [initial_state]
        for i in range(num_imgs):
            new_sys = initial_state.copy()
            new_pos = initial_state.get_positions() + ((i+1.)/(num_imgs+1.))*(final_state.get_positions()-initial_state.get_positions())
            new_sys.set_positions(new_pos)
            path.append(new_sys)
        path.append(final_state)
        return path

    #Creates a bisection path. After the first bisected image, others are created
    #in the direction of the lowest force (most uphill on PES).
    def create_bisection_path(self):
        #Creates the first bisected image
        if self.path_type =='linear' or self.path_type=='Linear':
            path = self.linear_interpolation(self.IS,self.FS,1)
        elif self.path_type =='idpp' or self.path_type == 'IDPP':
            path=[self.IS,self.IS.copy(),self.FS]
            neb_=NEB(path,k=0.15,climb=True)
            neb_.interpolate(method='idpp')
        '''elif self.path_type=='fit':
            from fitted_path import loocv_direct_func,loocv_func
            assert 'datafile' in self.kwargs
            if 'weight' not in self.kwargs:
                self.kwargs['weight']=0.
            if 'fit_type' not in self.kwargs:
                self.kwargs['fit_type']=3
            ads_inds = [a.index for a in self.IS if a.tag==0]
            lc = np.linalg.norm(self.IS.get_positions()[2]-self.IS.get_positions()[3])
            rc_i = np.linalg.norm(self.IS.get_positions()[ads_inds[0]]-self.IS.get_positions()[ads_inds[-1]])
            rc_f = np.linalg.norm(self.FS.get_positions()[ads_inds[0]]-self.FS.get_positions()[ads_inds[-1]])
            #target = (rc_i+rc_f)/2.
            #target/=lc
            target = 0.5
            target_list = [0.,0.5,1.]
            func = loocv_direct_func(self.IS,self.FS,self.kwargs.pop('datafile'),**self.kwargs)
            #func = loocv_func(self.IS,self.FS,**self.kwargs)
            nimg = func(target)
            #nimg = loocv_func(self.IS,self.FS,tar_rc,self.datafile,weight=0.,degree=1.,rel_rc=True)
            path = [self.IS,nimg,self.FS]
        '''#Create energy and force lists so they can be re-used if we go greedy
        old_energy_list=[]
        force_list=[]
        current_ind=1
        tangent =path[-1].get_positions()-path[0].get_positions()
        if self.verbose:
            og_forces=[]
            inits=[]
            #l_r=[]
            #l_r_forces=[]
            #l_r.append('m')
        if self.write:
            energy_inds=[0]
        #Iterate through the specified number of times
        for i in range(self.nimgs-1):
            current_system=path[current_ind]
            #Get a calc and set the appropriate directory
            self.attach_calculator(current_system)
            current_system.get_calculator().set(outdir=self.outdir+'_path')
            #Calculate and collect the forces and energies
            forces = current_system.get_forces()
            nrg = current_system.get_potential_energy()
            old_energy_list.insert(current_ind-1,nrg)
            #Get the forces in both directions along the reaction path
            left_tan = path[current_ind-1].get_positions() - path[current_ind].get_positions()
            left_tan/=np.sqrt(np.vdot(left_tan,left_tan))
            right_tan = path[current_ind+1].get_positions() - path[current_ind].get_positions()
            right_tan /=np.sqrt(np.vdot(right_tan,right_tan))
            lforce = np.vdot(forces,left_tan)
            rforce = np.vdot(forces,right_tan)
            #l_r_forces.append([lforce,rforce])
            #Collect the min force
            force_list.insert(current_ind-1,np.min([lforce,rforce]))
            if self.verbose:
                inits.insert(current_ind-1, current_system.get_positions())
                og_forces.insert(current_ind-1,forces)
            #Check for the correct direction and then create the new image
            if rforce<lforce:
                #l_r.append('r')
                if self.path_type =='linear' or self.path_type=='Linear':
                    nimg = self.linear_interpolation(current_system,path[current_ind+1],1)[1]
                elif self.path_type == 'IDPP' or self.path_type=='idpp':
                    nimg = current_system.copy()
                    p_=[current_system,nimg,path[current_ind+1]]
                    neb_=NEB(p_,k=0.15,climb=True)
                    neb_.interpolate(method='idpp')
                    #self.mod_idpp_interpolate(p_,traj=None,log=None,mic=False)
                    nimg = p_[1]
                '''elif self.path_type=='fit':
                    #rc_i = np.linalg.norm(current_system.get_positions()[ads_inds[0]]-current_system.get_positions()[ads_inds[-1]])
                    #rc_f = np.linalg.norm(path[current_ind+1].get_positions()[ads_inds[0]]-path[current_ind+1].get_positions()[ads_inds[-1]])
                    #target = (rc_i+rc_f)/2. 
                    #target/=lc
                    target = (target_list[current_ind]+target_list[current_ind+1])/2.
                '''#New image is to the 'right', so we must increment the index
                current_ind+=1
            else:
                #l_r.append('l')
                if self.path_type =='linear' or self.path_type=='Linear':
                    nimg = self.linear_interpolation(path[current_ind-1],current_system,1)[1]
                elif self.path_type == 'IDPP' or self.path_type=='idpp':
                    nimg=current_system.copy()
                    p_=[path[current_ind-1],nimg,current_system]
                    neb_=NEB(p_,k=0.15,climb=True)
                    neb_.interpolate(method='idpp')
                    #self.mod_idpp_interpolate(p_,traj=None,log=None,mic=False)
                    nimg=p_[1]
                '''elif self.path_type=='fit':
                    #rc_i = np.linalg.norm(path[current_ind-1].get_positions()[ads_inds[0]]-path[current_ind-1].get_positions()[ads_inds[-1]])   
                    #rc_f = np.linalg.norm(current_system.get_positions()[ads_inds[0]]-current_system.get_positions()[ads_inds[-1]])
                    #target = (rc_i+rc_f)/2.
                    #target/=lc
                    target = (target_list[current_ind]+target_list[current_ind-1])/2.
            if self.path_type=='fit':
                #nimg = loocv_func(self.IS,self.FS,tar_rc,self.datafile,weight=0.,degree=1.,rel_rc=True)
                nimg = func(target)
                target_list.insert(current_ind,target)'''
            path.insert(current_ind,nimg)
            if self.write:
                energy_inds.insert(current_ind-1,i+1)
        #TEMPORARY
        #pickle.dump(neb_,open('neb_object.pckl','wb'),-1)
        #pickle.dump([lforce,rforce,forces],open('idpp_diagnostic.pckl','wb'))
        #write_trajectory('idpp_images.traj',[current_system,nimg])
        #Put in final energy and force
        current_system = path[current_ind]
        self.attach_calculator(current_system)
        forces = current_system.get_forces()
        nrg = current_system.get_potential_energy()
        old_energy_list.insert(current_ind-1,nrg)
        left_tan = path[current_ind-1].get_positions() - path[current_ind].get_positions()
        left_tan/=np.sqrt(np.vdot(left_tan,left_tan))
        right_tan = path[current_ind+1].get_positions() - path[current_ind].get_positions()
        right_tan /=np.sqrt(np.vdot(right_tan,right_tan))
        lforce = np.vdot(forces,left_tan)
        rforce = np.vdot(forces,right_tan)
        #l_r_forces.append([lforce,rforce])
        force_list.insert(current_ind-1,np.min([lforce,rforce]))
        if self.verbose:
            og_forces.insert(current_ind-1,forces)
            inits.insert(current_ind-1,current_system.get_positions())
            self.og_forces=og_forces
            self.inits=inits
            #TEMP
            #self.lr = l_r
            #pickle.dump(self.lr,open('bisect_progress.pckl','wb'))
            #pickle.dump(l_r_forces,open('bisect_forces.pckl','wb'))
        if self.write:
            self.inds = [list(energy_inds).index(i) for i in range(self.nimgs)]
        return path,old_energy_list,force_list

    #Attaches calculators to images. Seems to work? Sketchy.
    def attach_calculator(self, image):
        ncalc = copy.copy(self.calc)
        image.set_calculator(ncalc)

    #Select the images to relax when being greedy.
    #Can use 'force' or 'energy' as metrics
    def select_images(self):
        #Check to see if we have previously calculated forces and energies
        #If not, get them
        if not hasattr(self,'energy_list') or not hasattr(self,'force_list'):
            self.energy_list=[]
            self.force_list=[]
            tangent = self.path[-1].get_positions()-self.path[0].get_positions()
            tangent/=np.sqrt(np.vdot(tangent,tangent))
            if self.verbose:
                self.tangent = tangent
                positions=[]
                og_forces=[]
            #Only care about the intermediate images
            for i,img in enumerate(self.path[1:-1]):
                forces = img.get_forces()
                nrg = img.get_potential_energy()
                if self.verbose:    
                    og_forces.append(forces)
                    positions.append(img.get_positions())
                if self.path_type=='linear' or self.path_type=='Linear':
                    tan_force = np.vdot(forces,tangent)
                    self.force_list.append(-1*np.abs(tan_force))
                #if self.path_type=='idpp' or self.path_type=='IDPP' or self.path_type=='fit':
                else:    
                    ltan = self.path[i].get_positions()-img.get_positions()
                    ltan /=np.sqrt(np.vdot(ltan,ltan))
                    rtan = self.path[i+2].get_positions()-img.get_positions()
                    rtan/=np.sqrt(np.vdot(rtan,rtan))
                    lforce = np.vdot(forces,ltan)
                    rforce = np.vdot(forces,rtan)
                    self.force_list.append(np.min([lforce,rforce]))
                self.energy_list.append(nrg)
            if self.verbose:
                self.inits = positions
                self.og_forces=og_forces
        #Now use our list and the appropriate metric to select the images
        rimages=[]
        if self.relax_num>0:
            if self.metric=='force':
                #Want lowest force
                inds = np.argsort(self.force_list)[:self.relax_num]
                inds = np.sort(inds)
                #rimages = list(np.array(path)[inds])
                for ind in inds:
                    rimages.append(self.path[ind+1])
            elif self.metric=='energy':
                #Want highest energy
                inds = np.argsort(self.energy_list)[-1*self.relax_num:]
                inds = np.sort(inds)
                #rimages=list(np.array(path)[inds])
                for ind in inds:
                    rimages.append(self.path[ind+1])
            else:
                raise ValueError('%s is not a valid choice for the metric. Please use either "force" or "energy"' % metric)
            #for i in range(len(rimages)):
            #    rimages[i] = Atoms(list(rimages[i]))
        return rimages

    #From ase. Modified to now be per image. I hope.
    #Seems to fail, so nevermind. need NEB forces or else all of the images are almost the same
    def mod_idpp_interpolate(self,path, traj='idpp.traj', log='idpp.log', fmax=0.1,
                         optimizer=BFGS, mic=False):
        d1 = path[0].get_all_distances(mic=mic)
        d2 = path[-1].get_all_distances(mic=mic)
        d = (d2 - d1) / (len(path) - 1)
        old = []
        for i, image in enumerate(path):
            old.append(image.calc)
            image.calc = IDPP(d1 + i * d, mic=mic)
            opt = optimizer(image, trajectory=traj, logfile=log)
            opt.run(fmax=fmax)
        for image, calc in zip(path, old):
            image.calc = calc


    #Set the appropriate constraints and then relax the images
    #Outputs trajectory files and pckl file of energies
    def relax_images(self):
        energies =[]
        if self.write:
            geosteps=[]
            times=[]
        #Set the appropriate constraints
        for i,image in enumerate(self.relax_imgs):
            ads_inds = [j for j,a in enumerate(image) if a.tag==0]
            #FBL
            if isinstance(self.rxn_coord,tuple):
                assert len(self.rxn_coord)==2
                constraints = [FixBondLength(self.rxn_coord[0],self.rxn_coord[1])] 
            #Relax perp to a vector
            elif isinstance(self.rxn_coord,np.ndarray):
                assert p.shape[0] ==len(image)
                assert p.shape[1] ==3 
                constraints=[FixedPlane(ai,self.rxn_coord[ai]) for ai in ads_inds]
            elif isinstance(self.rxn_coord,list):
                if isinstance(self.rxn_coord[0],list):
                    #Perp again
                    constraints=[FixedPlane(ai,self.rxn_coord[ai]) for ai in ads_inds]
                else:
                    #FBL again
                    constraints = [FixBondLength(self.rxn_coord[0],self.rxn_coord[1])]
            else:
                raise TypeError('The reaction coordinate must be a list, tuple, or numpy array')
            image.set_constraint(image.constraints+constraints)
            #Calc has to be reset. New outdir set as well.
            self.attach_calculator(image)
            image.get_calculator().set(outdir=self.outdir+'%i' % (i+1))
            #Optimizer and traj file is set and then we are off!
            opt = self.optimizer(image,logfile=self.logfile+'%i.log' % (i+1))
            traj = io.PickleTrajectory(self.traj +'%i.traj' % (i+1), 'w', image)
            opt.attach(traj)
            if self.write:
                c_,steps =opt.Aidan_run(fmax=0.02,threshold=0)
                geosteps.append(steps) 
            else:
                opt.run(fmax=0.02)
            energies.append(image.get_potential_energy()) 
        #Write the "TS" geometry to its own trajectory file
        write_trajectory(self.traj+'_TS.traj',[self.relax_imgs[np.argmax(energies)]])
        if self.write:
            self.geosteps=geosteps
        return energies            
    
    #Quick curve fitting method from ASE's NEB class
    def fit0(self,E, F, R, cell=None, pbc=None):
        """Constructs curve parameters from the NEB images."""
        E = np.array(E) - E[0]
        n = len(E)
        Efit = np.empty((n - 1) * 20 + 1)
        Sfit = np.empty((n - 1) * 20 + 1)

        s = [0]
        dR = np.zeros_like(R)
        for i in range(n):
            if i < n - 1:
                dR[i] = R[i + 1] - R[i]
                if cell is not None and pbc is not None:
                    dR[i], _ = find_mic(dR[i], cell, pbc)
                s.append(s[i] + sqrt((dR[i]**2).sum()))
            else:
                dR[i] = R[i] - R[i - 1]
                if cell is not None and pbc is not None:
                    dR[i], _ = find_mic(dR[i], cell, pbc)

        lines = []
        dEds0 = None
        for i in range(n):
            d = dR[i]
            if i == 0:
                ds = 0.5 * s[1]
            elif i == n - 1:
                ds = 0.5 * (s[-1] - s[-2])
            else:
                ds = 0.25 * (s[i + 1] - s[i - 1])

            d = d / sqrt((d**2).sum())
            dEds = -(F[i] * d).sum()
            x = np.linspace(s[i] - ds, s[i] + ds, 3)
            y = E[i] + dEds * (x - s[i])
            lines.append((x, y))

            if i > 0:
                s0 = s[i - 1]
                s1 = s[i]
                x = np.linspace(s0, s1, 20, endpoint=False)
                c = np.linalg.solve(np.array([(1, s0, s0**2, s0**3),
                                              (1, s1, s1**2, s1**3),
                                              (0, 1, 2 * s0, 3 * s0**2),
                                              (0, 1, 2 * s1, 3 * s1**2)]),
                                    np.array([E[i - 1], E[i], dEds0, dEds]))
                y = c[0] + x * (c[1] + x * (c[2] + x * c[3]))
                Sfit[(i - 1) * 20:i * 20] = x
                Efit[(i - 1) * 20:i * 20] = y

            dEds0 = dEds

        Sfit[-1] = s[-1]
        Efit[-1] = E[-1]
        return s, E, Sfit, Efit, lines

    #Gets the barrier from the Drag calculation. Based off of the method from ASE's NEBtools. 
    #If fit=True, the barrier is estimated based on the interpolated fit to the images; if fit=False,
    #the barrier is taken as the maximum energy image without interpolation.
    #Set raw=True to get the raw energy of the transition state instead of the forward barrier.
    def get_barrier(self,fit=False,raw=True):
        s, E, Sfit, Efit, lines = self.get_fit()
        dE = E[-1] - E[0]
        if fit:
            barrier = max(Efit)
        else:
            barrier = max(E)
        if raw:
            #Should this be relax_imgs instead? Do we have a calc for the IS?
            barrier += self.path[0].get_potential_energy()
        return barrier, dE

    #Makes a band plot on a matplotlib axes object, unless ax=None.
    #Adapeted from ASE's NEBtools
    def plot_band(self,ax=None):
        if not ax:
            from matplotlib import pyplot
            fig = pyplot.figure()
            ax = fig.add_subplot(111)
        else:
            fig = None
        s, E, Sfit, Efit, lines = self.get_fit()
        ax.plot(s, E, 'o')
        for x, y in lines:
            ax.plot(x, y, '-g')
        ax.plot(Sfit, Efit, 'k-')
        ax.set_xlabel('path [$\AA$]')
        ax.set_ylabel('energy [eV]')
        Ef = max(Efit) - E[0]
        Er = max(Efit) - E[-1]
        dE = E[-1] - E[0]
        ax.set_title('$E_\mathrm{f} \\approx$ %.3f eV; '
                     '$E_\mathrm{r} \\approx$ %.3f eV; '
                     '$\\Delta E$ = %.3f eV'
                     % (Ef, Er, dE))
        return fig

    #Used to create a fit to the Drag image results and return the results.
    #Adpated from ASE's NEBtools
    def get_fit(self):
        images = self.relax_imgs
        if not hasattr(images, 'repeat'):
            from ase.gui.images import Images
            images = Images(images)
        N = images.repeat.prod()
        natoms = images.natoms // N
        R = images.P[:, :natoms]
        E = images.E
        F = images.F[:, :natoms]
        s, E, Sfit, Efit, lines = fit0(E, F, R, images.A[0], images.pbc)
        return s, E, Sfit, Efit, lines


#From ase
class IDPP(Calculator):
    """Image dependent pair potential.

    See:

        Improved initial guess for minimum energy path calculations.

        Chem. Phys. 140, 214106 (2014)
    
"""
    implemented_properties = ['energy', 'forces']

    def __init__(self, target, mic):
        Calculator.__init__(self)
        self.target = target
        self.mic = mic

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        P = atoms.get_positions()
        d = []
        D = []
        for p in P:
            Di = P - p
            if self.mic:
                Di, di = find_mic(Di, atoms.get_cell(), atoms.get_pbc())
            else:
                di = np.sqrt((Di**2).sum(1))
            d.append(di)
            D.append(Di)
        d = np.array(d)
        D = np.array(D)

        dd = d - self.target
        d.ravel()[::len(d) + 1] = 1  # avoid dividing by zero
        d4 = d**4
        e = 0.5 * (dd**2 / d4).sum()
        f = -2 * ((dd * (1 - 2 * dd / d) / d**5)[..., np.newaxis] * D).sum(0)
        self.results = {'energy': e, 'forces': f}


