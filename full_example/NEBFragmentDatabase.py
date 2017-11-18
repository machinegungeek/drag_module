import numpy as np
from ase import Atoms
from ase.io import read
from ase.lattice.surface import *
from ase.constraints import FixAtoms
import os

from Database import Database

class NEBFragmentDatabase(Database):

    def __init__(self):
        self.initialize()

    def initialize(self):
        self.build_members()
        self.build_database()

    def build_members(self):
        self.members =['Ag_C-H', 'Ag_CH-H', 'Ag_CH2-H', 'Ag_CH3-H', 'Au_C-H', 'Au_CH-H', 'Au_CH2-H', 'Au_CH3-H', 'Cu_C-H', 'Cu_CH-H', 'Cu_CH2-H', 'Cu_CH3-H', 'Ni_C-H', 'Ni_CH-H', 'Ni_CH2-H', 'Ni_CH3-H', 'Pd_C-H', 'Pd_CH-H', 'Pd_CH2-H', 'Pd_CH3-H', 'Pt_C-H', 'Pt_CH-H', 'Pt_CH2-H', 'Pt_CH3-H', 'Re_C-H', 'Re_CH-H', 'Re_CH2-H', 'Re_CH3-H', 'Rh_C-H', 'Rh_CH-H', 'Rh_CH2-H', 'Rh_CH3-H', 'Ru_C-H', 'Ru_CH-H', 'Ru_CH2-H', 'Ru_CH3-H','Ag_N-H', 'Au_N-H', 'Cu_N-H', 'Ni_N-H', 'Pd_N-H', 'Pt_N-H', 'Re_N-H', 'Rh_N-H', 'Ru_N-H', 'Ag_NH-H', 'Au_NH-H', 'Cu_NH-H', 'Ni_NH-H', 'Pd_NH-H', 'Pt_NH-H', 'Re_NH-H', 'Rh_NH-H', 'Ru_NH-H', 'Ag_NH2-H', 'Au_NH2-H', 'Cu_NH2-H', 'Ni_NH2-H', 'Pd_NH2-H', 'Pt_NH2-H', 'Re_NH2-H', 'Rh_NH2-H', 'Ru_NH2-H','Ag_H-H', 'Au_H-H', 'Cu_H-H', 'Ni_H-H', 'Pd_H-H', 'Pt_H-H', 'Re_H-H', 'Rh_H-H', 'Ru_H-H', 'Ag_O-O', 'Au_O-O', 'Cu_O-O', 'Ni_O-O', 'Pd_O-O', 'Pt_O-O', 'Re_O-O', 'Rh_O-O', 'Ru_O-O', 'Ag_N-N', 'Au_N-N', 'Cu_N-N', 'Ni_N-N', 'Pd_N-N', 'Pt_N-N', 'Re_N-N', 'Rh_N-N', 'Ru_N-N','Ag_C-O', 'Au_C-O', 'Cu_C-O', 'Ni_C-O', 'Pd_C-O', 'Pt_C-O', 'Re_C-O', 'Rh_C-O', 'Ru_C-O', 'Ag_CO-O', 'Au_CO-O', 'Cu_CO-O', 'Ni_CO-O', 'Pd_CO-O', 'Pt_CO-O', 'Re_CO-O', 'Rh_CO-O', 'Ru_CO-O', 'Ag_N-O', 'Au_N-O', 'Cu_N-O', 'Ni_N-O', 'Pd_N-O', 'Pt_N-O', 'Re_N-O', 'Rh_N-O', 'Ru_N-O']
        self.expected_members_len=117

    def build_database(self):
        self.data ={}
        #Build the member database. Have to encode the initial and final fragments, as well as the appropriate adsorption site.
        #Most fragments have to also have an orientation label, ie XY_up or XY_down, etc.
        #Finally, to make the interpolation work, we may need to interchange indices (in the form of [[old],[new]]) or even make translations (in units of the lattice value)
        for m in self.members:
            self.data[m] = {'name':m,'metal':m.split('_')[0],'initial_interchanges':[],'final_interchanges':[]}
        self.data['Ag_C-H'].update({'gas':'CH','initial_fragments':['CH_up'],'initial_sites':['hcp'],'final_fragments':['C','H'],'final_sites':['hcp','hcp2']})
        self.data['Cu_C-H'].update({'gas':'CH','initial_fragments':['CH_up'],'initial_sites':['hcp'],'final_fragments':['C','H'],'final_sites':['hcp','hcp2']})
        self.data['Rh_C-H'].update({'gas':'CH','initial_fragments':['CH_up'],'initial_sites':['hcp'],'final_fragments':['C','H'],'final_sites':['hcp','hcp2']})
        self.data['Ru_C-H'].update({'gas':'CH','initial_fragments':['CH_up'],'initial_sites':['hcp'],'final_fragments':['C','H'],'final_sites':['hcp','hcp2']})
        self.data['Pt_C-H'].update({'gas':'CH','initial_fragments':['CH_up'],'initial_sites':['hcp'],'final_fragments':['C','H'],'final_sites':['hcp','hcp2']})
        self.data['Au_C-H'].update({'gas':'CH','initial_fragments':['CH_up'],'initial_sites':['fcc'],'final_fragments':['C','H'],'final_sites':['fcc','fcc2']})
        self.data['Pd_C-H'].update({'gas':'CH','initial_fragments':['CH_up'],'initial_sites':['fcc'],'final_fragments':['C','H'],'final_sites':['fcc','fcc2']})
        self.data['Ni_C-H'].update({'gas':'CH','initial_fragments':['CH_tilt'],'initial_sites':['hcp'],'final_fragments':['C','H'],'final_sites':['hcp','hcp2']})
        self.data['Re_C-H'].update({'gas':'CH','initial_fragments':['CH_tilt'],'initial_sites':['hcp'],'final_fragments':['C','H'],'final_sites':['hcp','hcp2']})
        self.data['Ag_CH-H'].update({'gas':'CH2','initial_fragments':['CH2_flat'],'initial_sites':['hcp'],'final_fragments':['CH_up','H'],'final_sites':['hcp','hcp2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Re_CH-H'].update({'gas':'CH2','initial_fragments':['CH2_flat'],'initial_sites':['hcp'],'final_fragments':['CH_up','H'],'final_sites':['hcp','fcc']})
        self.data['Ru_CH-H'].update({'gas':'CH2','initial_fragments':['CH2_flat'],'initial_sites':['hcp'],'final_fragments':['CH_up','H'],'final_sites':['hcp','fcc']})
        self.data['Ni_CH-H'].update({'gas':'CH2','initial_fragments':['CH2_flat'],'initial_sites':['fcc'],'final_fragments':['CH_tilt','H'],'final_sites':['hcp','hcp2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Pt_CH-H'].update({'gas':'CH2','initial_fragments':['CH2_flat'],'initial_sites':['hcp'],'final_fragments':['CH_up','H'],'final_sites':['fcc','ontop']})
        self.data['Au_CH-H'].update({'gas':'CH2','initial_fragments':['CH2_down'],'initial_sites':['bridge'],'final_fragments':['CH_up','H'],'final_sites':['fcc','fcc2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Pd_CH-H'].update({'gas':'CH2','initial_fragments':['CH2_down'],'initial_sites':['bridge'],'final_fragments':['CH_up','H'],'final_sites':['fcc','fcc2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Cu_CH-H'].update({'gas':'CH2','initial_fragments':['CH2_down'],'initial_sites':['fcc'],'final_fragments':['CH_up','H'],'final_sites':['hcp','hcp2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Rh_CH-H'].update({'gas':'CH2','initial_fragments':['CH2_down'],'initial_sites':['fcc'],'final_fragments':['CH_up','H'],'final_sites':['hcp','fcc']})#,'initial_interchanges':[[5,6],[6,5]]})
        self.data['Ru_CH2-H'].update({'gas':'CH3','initial_fragments':['CH3_flat'],'initial_sites':['hcp'],'final_fragments':['CH2_flat','H'],'final_sites':['hcp','hcp2']})
        self.data['Ni_CH2-H'].update({'gas':'CH3','initial_fragments':['CH3_flat'],'initial_sites':['bridge'],'final_fragments':['CH2_flat','H'],'final_sites':['hcp','fcc']})#,'initial_interchanges':[[5,6],[6,5]]})
        self.data['Ag_CH2-H'].update({'gas':'CH3','initial_fragments':['CH3_flat'],'initial_sites':['bridge'],'final_fragments':['CH2_flat','H'],'final_sites':['fcc2','fcc']})
        self.data['Cu_CH2-H'].update({'gas':'CH3','initial_fragments':['CH3_flat'],'initial_sites':['bridge'],'final_fragments':['CH2_flat','H'],'final_sites':['hcp','hcp2'],'initial_interchanges':[[7,5,6],[5,6,7]]})
        self.data['Re_CH2-H'].update({'gas':'CH3','initial_fragments':['CH3_flat'],'initial_sites':['bridge'],'final_fragments':['CH2_flat','H'],'final_sites':['hcp','fcc']})#,'initial_interchanges':[[5,6],[6,5]]})
        self.data['Pt_CH2-H'].update({'gas':'CH3','initial_fragments':['CH3_flat'],'initial_sites':['ontop'],'final_fragments':['CH2_flat','H'],'final_sites':['hcp2','ontop']})
        self.data['Au_CH2-H'].update({'gas':'CH3','initial_fragments':['CH3_flat'],'initial_sites':['ontop2'],'final_fragments':['CH2_down','H'],'final_sites':['bridge2','fcc']})
        self.data['Pd_CH2-H'].update({'gas':'CH3','initial_fragments':['CH3_flat'],'initial_sites':['ontop2'],'final_fragments':['CH2_down','H'],'final_sites':['bridge2','fcc']})
        self.data['Rh_CH2-H'].update({'gas':'CH3','initial_fragments':['CH3_flat'],'initial_sites':['ontop2'],'final_fragments':['CH2_down','H'],'final_sites':['fcc2','fcc']})
        self.data['Ru_CH3-H'].update({'gas':'CH4','initial_fragments':['CH4'],'initial_sites':['fcc'],'final_fragments':['CH3_flat','H'],'final_sites':['hcp','fcc']})
        self.data['Pt_CH3-H'].update({'gas':'CH4','initial_fragments':['CH4'],'initial_sites':['fcc'],'final_fragments':['CH3_flat','H'],'final_sites':['ontop','ontop2']})#,'initial_interchanges':[[5,6,7,8],[8,7,6,5]]})
        self.data['Ni_CH3-H'].update({'gas':'CH4','initial_fragments':['CH4'],'initial_sites':['fcc'],'final_fragments':['CH3_flat','H'],'final_sites':['bridge','fcc2']})#,'initial_interchanges':[[5,6,7,8],[8,7,6,5]]})
        self.data['Ag_CH3-H'].update({'gas':'CH4','initial_fragments':['CH4'],'initial_sites':['fcc'],'final_fragments':['CH3_flat','H'],'final_sites':['bridge','hcp']})#,'initial_interchanges':[[5,6,7,8],[8,7,6,5]]})
        self.data['Cu_CH3-H'].update({'gas':'CH4','initial_fragments':['CH4'],'initial_sites':['fcc'],'final_fragments':['CH3_flat','H'],'final_sites':['bridge','fcc2']})#,'initial_interchanges':[[5,6,7,8],[8,7,6,5]]})
        self.data['Re_CH3-H'].update({'gas':'CH4','initial_fragments':['CH4'],'initial_sites':['fcc'],'final_fragments':['CH3_flat','H'],'final_sites':['bridge','hcp']})#,'initial_interchanges':[[5,6,7,8],[8,7,6,5]]})
        self.data['Au_CH3-H'].update({'gas':'CH4','initial_fragments':['CH4'],'initial_sites':['fcc'],'final_fragments':['CH3_flat','H'],'final_sites':['ontop','fcc']})#,'initial_interchanges':[[5,6,7,8],[8,7,6,5]]})
        self.data['Pd_CH3-H'].update({'gas':'CH4','initial_fragments':['CH4'],'initial_sites':['fcc'],'final_fragments':['CH3_flat','H'],'final_sites':['ontop','fcc']})#,'initial_interchanges':[[5,6,7,8],[8,7,6,5]]})
        self.data['Rh_CH3-H'].update({'gas':'CH4','initial_fragments':['CH4'],'initial_sites':['fcc'],'final_fragments':['CH3_flat','H'],'final_sites':['ontop','fcc']})#,'initial_interchanges':[[5,6,7,8],[8,7,6,5]]})
        self.data['Ag_N-H'].update({'gas':'NH','initial_fragments':['NH_up'],'initial_sites':['fcc'],'final_fragments':['N','H'],'final_sites':['fcc','fcc2']})
        self.data['Au_N-H'].update({'gas':'NH','initial_fragments':['NH_up'],'initial_sites':['fcc'],'final_fragments':['N','H'],'final_sites':['fcc','fcc2']})
        self.data['Pd_N-H'].update({'gas':'NH','initial_fragments':['NH_up'],'initial_sites':['fcc'],'final_fragments':['N','H'],'final_sites':['fcc','fcc2']})
        self.data['Rh_N-H'].update({'gas':'NH','initial_fragments':['NH_up'],'initial_sites':['fcc'],'final_fragments':['N','H'],'final_sites':['fcc','fcc2']})
        self.data['Pt_N-H'].update({'gas':'NH','initial_fragments':['NH_up'],'initial_sites':['fcc'],'final_fragments':['N','H'],'final_sites':['fcc','fcc2']})
        self.data['Cu_N-H'].update({'gas':'NH','initial_fragments':['NH_tilt'],'initial_sites':['fcc'],'final_fragments':['N','H'],'final_sites':['fcc','fcc2']})
        self.data['Ni_N-H'].update({'gas':'NH','initial_fragments':['NH_tilt'],'initial_sites':['fcc'],'final_fragments':['N','H'],'final_sites':['fcc','fcc2']})
        self.data['Ru_N-H'].update({'gas':'NH','initial_fragments':['NH_tilt'],'initial_sites':['hcp'],'final_fragments':['N','H'],'final_sites':['hcp','hcp2']})
        self.data['Re_N-H'].update({'gas':'NH','initial_fragments':['NH_up'],'initial_sites':['hcp'],'final_fragments':['N','H'],'final_sites':['hcp','hcp2']})
        self.data['Ag_NH-H'].update({'gas':'NH2','initial_fragments':['NH2_down'],'initial_sites':['bridge'],'final_fragments':['NH_up','H'],'final_sites':['fcc','fcc2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Au_NH-H'].update({'gas':'NH2','initial_fragments':['NH2_down'],'initial_sites':['bridge'],'final_fragments':['NH_up','H'],'final_sites':['fcc','fcc2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Pd_NH-H'].update({'gas':'NH2','initial_fragments':['NH2_down'],'initial_sites':['bridge'],'final_fragments':['NH_up','H'],'final_sites':['fcc','fcc2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Rh_NH-H'].update({'gas':'NH2','initial_fragments':['NH2_down'],'initial_sites':['bridge'],'final_fragments':['NH_up','H'],'final_sites':['fcc','fcc2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Pt_NH-H'].update({'gas':'NH2','initial_fragments':['NH2_down'],'initial_sites':['bridge'],'final_fragments':['NH_up','H'],'final_sites':['fcc','fcc2']})
        self.data['Re_NH-H'].update({'gas':'NH2','initial_fragments':['NH2_down'],'initial_sites':['bridge'],'final_fragments':['NH_up','H'],'final_sites':['hcp','hcp2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Ru_NH-H'].update({'gas':'NH2','initial_fragments':['NH2_down'],'initial_sites':['bridge'],'final_fragments':['NH_tilt','H'],'final_sites':['hcp','hcp2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Ni_NH-H'].update({'gas':'NH2','initial_fragments':['NH2_down'],'initial_sites':['bridge'],'final_fragments':['NH_tilt','H'],'final_sites':['fcc','fcc2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Cu_NH-H'].update({'gas':'NH2','initial_fragments':['NH2_down'],'initial_sites':['bridge'],'final_fragments':['NH_tilt','H'],'final_sites':['fcc','fcc2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Ag_NH2-H'].update({'gas':'NH3','initial_fragments':['NH3_flat'],'initial_sites':['ontop2'],'final_fragments':['NH2_down','H'],'final_sites':['bridge2','fcc']})
        self.data['Au_NH2-H'].update({'gas':'NH3','initial_fragments':['NH3_flat'],'initial_sites':['ontop2'],'final_fragments':['NH2_down','H'],'final_sites':['bridge2','fcc']})
        self.data['Cu_NH2-H'].update({'gas':'NH3','initial_fragments':['NH3_flat'],'initial_sites':['ontop2'],'final_fragments':['NH2_down','H'],'final_sites':['bridge2','fcc']})
        self.data['Ni_NH2-H'].update({'gas':'NH3','initial_fragments':['NH3_flat'],'initial_sites':['ontop2'],'final_fragments':['NH2_down','H'],'final_sites':['bridge2','fcc']})
        self.data['Pd_NH2-H'].update({'gas':'NH3','initial_fragments':['NH3_flat'],'initial_sites':['ontop2'],'final_fragments':['NH2_down','H'],'final_sites':['bridge2','fcc']})
        self.data['Re_NH2-H'].update({'gas':'NH3','initial_fragments':['NH3_flat'],'initial_sites':['ontop2'],'final_fragments':['NH2_down','H'],'final_sites':['bridge2','fcc']})
        self.data['Rh_NH2-H'].update({'gas':'NH3','initial_fragments':['NH3_flat'],'initial_sites':['ontop2'],'final_fragments':['NH2_down','H'],'final_sites':['bridge2','fcc']})
        self.data['Ru_NH2-H'].update({'gas':'NH3','initial_fragments':['NH3_flat'],'initial_sites':['ontop2'],'final_fragments':['NH2_down','H'],'final_sites':['bridge2','fcc']})
        self.data['Pt_NH2-H'].update({'gas':'NH3','initial_fragments':['NH3_flat'],'initial_sites':['ontop2'],'final_fragments':['NH2_down','H'],'final_sites':['bridge2','bridge']})
        self.data['Pt_H-H'].update({'gas':'H2','initial_fragments':['H2_tilt'],'initial_sites':['ontop2'],'final_fragments':['H','H'],'final_sites':['ontop','ontop2']})
        self.data['Pd_H-H'].update({'gas':'H2','initial_fragments':['H2_tilt'],'initial_sites':['ontop2'],'final_fragments':['H','H'],'final_sites':['fcc','fcc2']})
        self.data['Rh_H-H'].update({'gas':'H2','initial_fragments':['H2_tilt'],'initial_sites':['hcp2'],'final_fragments':['H','H'],'final_sites':['fcc','fcc2']})
        self.data['Au_H-H'].update({'gas':'H2','initial_fragments':['H2_tilt'],'initial_sites':['fcc'],'final_fragments':['H','H'],'final_sites':['fcc','fcc2']})
        self.data['Ni_H-H'].update({'gas':'H2','initial_fragments':['H2_tilt'],'initial_sites':['fcc'],'final_fragments':['H','H'],'final_sites':['fcc','fcc2']})
        self.data['Re_H-H'].update({'gas':'H2','initial_fragments':['H2_tilt'],'initial_sites':['fcc'],'final_fragments':['H','H'],'final_sites':['fcc','fcc2']})
        self.data['Ru_H-H'].update({'gas':'H2','initial_fragments':['H2_tilt'],'initial_sites':['fcc'],'final_fragments':['H','H'],'final_sites':['fcc','fcc2']})
        self.data['Cu_H-H'].update({'gas':'H2','initial_fragments':['H2_tilt'],'initial_sites':['fcc'],'final_fragments':['H','H'],'final_sites':['hcp','hcp2']})
        self.data['Ag_H-H'].update({'gas':'H2','initial_fragments':['H2_tilt'],'initial_sites':['fcc'],'final_fragments':['H','H'],'final_sites':['hcp','hcp2']})
        self.data['Ag_O-O'].update({'gas':'O2','initial_fragments':['O2_up'],'initial_sites':['fcc'],'final_fragments':['O','O'],'final_sites':['fcc','fcc2']}) 
        self.data['Cu_O-O'].update({'gas':'O2','initial_fragments':['O2_up'],'initial_sites':['fcc'],'final_fragments':['O','O'],'final_sites':['fcc','fcc2']})
        self.data['Au_O-O'].update({'gas':'O2','initial_fragments':['O2_up'],'initial_sites':['ontop2'],'final_fragments':['O','O'],'final_sites':['fcc','fcc2']})
        self.data['Rh_O-O'].update({'gas':'O2','initial_fragments':['O2_up'],'initial_sites':['ontop2'],'final_fragments':['O','O'],'final_sites':['fcc','fcc2'],'initial_interchanges':[[4,5],[5,4]]})
        self.data['Pd_O-O'].update({'gas':'O2','initial_fragments':['O2_up'],'initial_sites':['ontop2'],'final_fragments':['O','O'],'final_sites':['fcc','fcc2']})
        self.data['Pt_O-O'].update({'gas':'O2','initial_fragments':['O2_up'],'initial_sites':['ontop2'],'final_fragments':['O','O'],'final_sites':['fcc','fcc2'],'initial_interchanges':[[4,5],[5,4]]})
        self.data['Re_O-O'].update({'gas':'O2','initial_fragments':['O2_tilt'],'initial_sites':['hcp'],'final_fragments':['O','O'],'final_sites':['hcp','hcp2']})
        self.data['Ru_O-O'].update({'gas':'O2','initial_fragments':['O2_tilt'],'initial_sites':['hcp'],'final_fragments':['O','O'],'final_sites':['hcp','hcp2']})
        self.data['Ni_O-O'].update({'gas':'O2','initial_fragments':['O2_tilt'],'initial_sites':['fcc'],'final_fragments':['O','O'],'final_sites':['fcc','fcc2']})
        self.data['Cu_N-N'].update({'gas':'N2','initial_fragments':['N2_tilt'],'initial_sites':['ontop2'],'final_fragments':['N','N'],'final_sites':['fcc','fcc2']})    
        self.data['Ni_N-N'].update({'gas':'N2','initial_fragments':['N2_tilt'],'initial_sites':['ontop2'],'final_fragments':['N','N'],'final_sites':['fcc','fcc2']})
        self.data['Pd_N-N'].update({'gas':'N2','initial_fragments':['N2_tilt'],'initial_sites':['ontop2'],'final_fragments':['N','N'],'final_sites':['fcc','fcc2']})
        self.data['Pt_N-N'].update({'gas':'N2','initial_fragments':['N2_tilt'],'initial_sites':['ontop2'],'final_fragments':['N','N'],'final_sites':['fcc','fcc2']})
        self.data['Rh_N-N'].update({'gas':'N2','initial_fragments':['N2_tilt'],'initial_sites':['ontop2'],'final_fragments':['N','N'],'final_sites':['fcc','fcc2']})
        self.data['Re_N-N'].update({'gas':'N2','initial_fragments':['N2_tilt'],'initial_sites':['ontop2'],'final_fragments':['N','N'],'final_sites':['hcp','hcp2']})        
        self.data['Ru_N-N'].update({'gas':'N2','initial_fragments':['N2_tilt'],'initial_sites':['ontop2'],'final_fragments':['N','N'],'final_sites':['hcp','hcp2']})
        self.data['Ag_N-N'].update({'gas':'N2','initial_fragments':['N2_tilt'],'initial_sites':['bridge'],'final_fragments':['N','N'],'final_sites':['fcc','fcc2']})        
        self.data['Au_N-N'].update({'gas':'N2','initial_fragments':['N2_tilt'],'initial_sites':['bridge'],'final_fragments':['N','N'],'final_sites':['fcc','fcc2']})
        self.data['Ag_C-O'].update({'gas':'CO','initial_fragments':['CO_down'],'initial_sites':['hcp'], 'final_fragments':['C','O'],'final_sites':['fcc','fcc2']})
        self.data['Ru_C-O'].update({'gas':'CO','initial_fragments':['CO_down'],'initial_sites':['hcp2'], 'final_fragments':['C','O'],'final_sites':['hcp2','hcp'],'initial_translations':[1.,1.],'final_translations':[1.,-1.]})
        self.data['Cu_C-O'].update({'gas':'CO','initial_fragments':['CO_down'],'initial_sites':['fcc'], 'final_fragments':['C','O'],'final_sites':['fcc','fcc2']})
        self.data['Pd_C-O'].update({'gas':'CO','initial_fragments':['CO_down'],'initial_sites':['fcc'], 'final_fragments':['C','O'],'final_sites':['fcc','fcc2']})
        self.data['Ni_C-O'].update({'gas':'CO','initial_fragments':['CO_down'],'initial_sites':['fcc'], 'final_fragments':['C','O'],'final_sites':['fcc','fcc2'],'final_translations':[-1.,1.]})
        self.data['Au_C-O'].update({'gas':'CO','initial_fragments':['CO_down'],'initial_sites':['bridge'], 'final_fragments':['C','O'],'final_sites':['fcc2','bridge']})
        self.data['Pt_C-O'].update({'gas':'CO','initial_fragments':['CO_down'],'initial_sites':['ontop2'], 'final_fragments':['C','O'],'final_sites':['fcc','fcc2']})
        self.data['Re_C-O'].update({'gas':'CO','initial_fragments':['CO_down'],'initial_sites':['ontop2'], 'final_fragments':['C','O'],'final_sites':['hcp','hcp2']})        
        self.data['Rh_C-O'].update({'gas':'CO','initial_fragments':['CO_down'],'initial_sites':['ontop2'], 'final_fragments':['C','O'],'final_sites':['fcc','fcc2']})    
        self.data['Ag_CO-O'].update({'gas':'CO2','initial_fragments':['CO2_flat'],'initial_sites':['hcp'], 'final_fragments':['CO_down','O'],'final_sites':['hcp','hcp2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Au_CO-O'].update({'gas':'CO2','initial_fragments':['CO2_flat'],'initial_sites':['hcp'], 'final_fragments':['CO_down','O'],'final_sites':['bridge','bridge2'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Cu_CO-O'].update({'gas':'CO2','initial_fragments':['CO2_flat'],'initial_sites':['bridge'], 'final_fragments':['CO_down','O'],'final_sites':['fcc','fcc2'],'initial_interchanges':[[5,6],[6,5]],'final_translations':[1.,1.,-1.]})
        self.data['Re_CO-O'].update({'gas':'CO2','initial_fragments':['CO2_flat'],'initial_sites':['bridge2'], 'final_fragments':['CO_down','O'],'final_sites':['ontop','hcp'],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Rh_CO-O'].update({'gas':'CO2','initial_fragments':['CO2_up'],'initial_sites':['hcp'], 'final_fragments':['CO_down','O'],'final_sites':['fcc2','fcc'],'final_translations':[1.,1.,-1.]})#'initial_interchanges':[[5,6],[6,5]],'final_translations':[1.,1.,-1.]})
        self.data['Pd_CO-O'].update({'gas':'CO2','initial_fragments':['CO2_up'],'initial_sites':['bridge'], 'final_fragments':['CO_down','O'],'final_sites':['fcc2','fcc'],'final_translations':[1.,1.,-1.]})#'initial_interchanges':[[5,6],[6,5]],'final_translations':[1.,1.,-1.]})
        self.data['Ru_CO-O'].update({'gas':'CO2','initial_fragments':['CO2_flat'],'initial_sites':['ontop'], 'final_fragments':['CO_down','O'],'final_sites':['hcp2','hcp'],'final_translations':[0.,0.,-2.],'initial_interchanges':[[5,6],[6,5]]})
        self.data['Ni_CO-O'].update({'gas':'CO2','initial_fragments':['CO2_flat'],'initial_sites':['ontop2'], 'final_fragments':['CO_down','O'],'final_sites':['fcc2','fcc']})
        self.data['Pt_CO-O'].update({'gas':'CO2','initial_fragments':['CO2_flat'],'initial_sites':['ontop'], 'final_fragments':['CO_down','O'],'final_sites':['ontop','ontop2'],'final_translations':[1.,1.,-1.]})#'initial_translations':[-1.,-1.,-1.],'final_translations':[-1.,-1,1.]})#,'initial_interchanges':[[5,6],[6,5]]})
        self.data['Ag_N-O'].update({'gas':'NO','initial_fragments':['NO_down'],'initial_sites':['fcc'],'final_fragments':['N','O'],'final_sites':['fcc','fcc2']})
        self.data['Cu_N-O'].update({'gas':'NO','initial_fragments':['NO_down'],'initial_sites':['fcc'],'final_fragments':['N','O'],'final_sites':['fcc','fcc2']})
        self.data['Ni_N-O'].update({'gas':'NO','initial_fragments':['NO_dtilt'],'initial_sites':['hcp'],'final_fragments':['N','O'],'final_sites':['fcc2','fcc']})
        self.data['Pd_N-O'].update({'gas':'NO','initial_fragments':['NO_down'],'initial_sites':['fcc'],'final_fragments':['N','O'],'final_sites':['fcc','fcc2']})
        self.data['Pt_N-O'].update({'gas':'NO','initial_fragments':['NO_down'],'initial_sites':['fcc'],'final_fragments':['N','O'],'final_sites':['fcc','fcc2']})        
        self.data['Re_N-O'].update({'gas':'NO','initial_fragments':['NO_down'],'initial_sites':['hcp'],'final_fragments':['N','O'],'final_sites':['hcp2','hcp'],'initial_translations':[1.,1.]})
        self.data['Ru_N-O'].update({'gas':'NO','initial_fragments':['NO_down'],'initial_sites':['hcp'],'final_fragments':['N','O'],'final_sites':['hcp','hcp2']})
        self.data['Rh_N-O'].update({'gas':'NO','initial_fragments':['NO_down'],'initial_sites':['hcp'],'final_fragments':['N','O'],'final_sites':['fcc2','fcc']})
        self.data['Au_N-O'].update({'gas':'NO','initial_fragments':['NO_down'],'initial_sites':['ontop2'],'final_fragments':['N','O'],'final_sites':['fcc','fcc2']})

    #The code to actually grab the list of images for a calculation.
    #test_neb = True returns a neb object (to check interpolation) instead of a list of images
    #prelim = True means we need to re-relax the initial and final structures based on my earlier fragment tests. For actual nebs, use False
    #zcheck allows IS and FS's where the adsorbate is basically desorbed to be brought closer for better NEB convergence
    #method is the path method. '' is basically 'linear'. 'reuse' is the special guy, where we re-use the path from a previous calculation (a glorified resubmit...)
    #directory is the directory where the struc_dirs, nebs, etc are located (basically 'NEBRun')
    #int_directory is the directory where we search for the images we want to reuse for method='reuse'
    #struc_dirs is where we are going to look for the initial and final states, given that we care (ie prelim=False)
    def get_system(self,name,intermediate_images=5,directory='',vacuum=6.0,test_neb=False,prelim=True,zcheck=False,method='',int_directory=None,start_layers=2,add_layers=0,fixed_layers=2,struc_dirs=['InitialStructures','FinalStructures']):
        #For calculating the true IS and FS
        if prelim:
            initial = self.get_structure(self.data[name]['metal'],self.data[name]['initial_fragments'],self.data[name]['initial_sites'],directory,vacuum,self.data[name]['initial_interchanges'],prelim=prelim,layers=start_layers,add_layers=add_layers,fixed_layers=fixed_layers)
            final = self.get_structure(self.data[name]['metal'],self.data[name]['final_fragments'],self.data[name]['final_sites'],directory,vacuum,self.data[name]['final_interchanges'],prelim=prelim,layers=start_layers,add_layers=add_layers,fixed_layers=fixed_layers)
        else:
            #If we have IS and FS structures, get them, applying translations, interchanges, etc as needed
            translations=None
            if self.data[name].has_key('initial_translations'):
                translations = self.data[name]['initial_translations']
            initial = self.get_structure(self.data[name]['metal'],self.data[name]['initial_fragments'],self.data[name]['initial_sites'],directory+'/%s' % struc_dirs[0],vacuum,self.data[name]['initial_interchanges'],prelim=prelim,path=name,translations=translations,layers=start_layers,add_layers=add_layers,fixed_layers=fixed_layers)
            translations=None
            if self.data[name].has_key('final_translations'):
                translations = self.data[name]['final_translations']
            final = self.get_structure(self.data[name]['metal'],self.data[name]['final_fragments'],self.data[name]['final_sites'],directory+'/%s' % struc_dirs[1],vacuum,self.data[name]['final_interchanges'],prelim=prelim,path=name,translations=translations,layers=start_layers,add_layers=add_layers, fixed_layers=fixed_layers)
        #Modify the heights if necessary and wanted. Basically put the adsorbate 1.9 ang away 
        if zcheck:
            z_dists=[]
            ads_inds = [i for i in range(len(initial)) if initial[i].tag==0]
            #Get z-distances of each adsorbate atom to one of the top metal atoms
            for i in ads_inds:
                z_dists.append(initial[i].position[-1]-initial[-1*len(ads_inds)-1].position[-1])
            z_dist = np.min(z_dists)
            if z_dist>2.2:
                #trans_d = (z_dist-1.)/2.
                trans_d = z_dist-1.9
                for j in ads_inds:
                    initial[j].position[-1]-=trans_d
        images = [initial]
        #Get the images from NEB's built in methods
        if method =='linear' or method=='idpp':
            images+=[initial.copy() for i in range(intermediate_images)]
            images.append(final)
            from ase.neb2 import NEB
            neb=NEB(images)
            neb.interpolate(method=method)
            return neb.images
        
        #Try to use old data. Messy :(
        elif method=='reuse':
            assert int_directory
            ok =True
            counter=0
            bcell=initial.cell
            #Load images one by one from int_directory, assuming they are numbered, until we run out of images to add. Add and fix the appropriate number of layers
            while ok:
                counter+=1
                try:
                    int_image = self.get_structure('',[''],[''],int_directory,vacuum,prelim=False,path=name,neb_name='_neb%i' % counter,layers=start_layers,add_layers=add_layers,fixed_layers=fixed_layers)
                    int_image.set_cell(bcell)
                    lineup = initial[0].position-int_image[0].position
                    int_image.positions+=lineup
                    images.append(int_image)
                except:
                    ok=False
            #Equalize all of the cells
            final.set_cell(bcell)
            images.append(final)
            return images
        #Just returns a bunch of copies. Counting on the template to do the interpolation then
        elif method=='':
            images+=[initial.copy() for i in range(intermediate_images)]
            images.append(final)

        if test_neb:
            from ase.neb import NEB
            neb = NEB(images,k=0.2,climb=True)
            neb.interpolate()
            return neb

        return images

    #Actually retrieves/creates the appropriate individual structures as requested by get_system()
    def get_structure(self,metal,fragments,sites,directory,vacuum=6.0,interchanges=[],prelim=True,path='',neb_name='',translations=None,add_layers=0,fixed_layers=2,layers=2):
        from SurfaceFragmentDatabase import SurfaceFragmentDatabase as sfd
        assert len(fragments)==len(sites)
        interchanges = list(np.array(interchanges)+2*add_layers)
        #If we need to calculate the IS/FS
        if prelim:
            if not directory:
                #Get it from the database
                return sfd().get_system(metal+'_fcc111',adsorbate=fragments,site=sites,layers=layers)
            #Get it from previous calcs, prbly in newFragmentTest or w/e
            #Metal base from db and then place the adsorbates based on the calculations
            else:
                #try:
                   #base = read(os.path.join(directory,'_'.join(fragments),metal+'_fcc111','%s_fcc111s.traj' % (metal)))
                base = sfd().get_system(metal+'_fcc111',layers=layers)
                for i in range(len(sites)):
                    site_trans=False
                    file_exist=False
                    #For each site try to find the appropriate .traj file
                    try:
                        struc=read(os.path.join(directory,fragments[i],metal+'_fcc111','%s_fcc111_%s_%s.traj' % (metal,fragments[i],sites[i])))
                        file_exist=True
                    except:
                        try:
                            struc=read(os.path.join(directory,fragments[i],metal+'_fcc111','%s_fcc111_%s_%s.traj' % (metal,fragments[i],sites[i][:-1])))
                            if sites[i][-1]=='2':
                                site_trans=True
                            file_exist=True
                        except:
                            struc = sfd().get_system(metal+'_fcc111',adsorbate=fragments[i],site=sites[i],layers=layers)
                    #If it worked, set stuff up properly
                    if file_exist:
                        (t1,t2,t3)=tuple(np.array((.38364,0.08574,1.18675))*-6)
                        struc.translate([t1,t2,t3])
                    frag = struc[4:len(struc)]
                    #I think this is a hard-coded fix for CO2. Bleh.. 
                    if list(frag.numbers)==[8,6]:
                        frag.numbers=[6,8]
                        frag.positions=frag.positions[[1,0]]
                    #Needed if site is set to xyz2. Moves it over two atoms in the x direction
                    if site_trans:
                        frag.translate([base.cell[0][0]/2.,0.,0.])
                    base=base+frag
        #Now we are concerned with NEBs
        #Get base file from dir/frag_frag/name/name_suf.traj. Should be in some form of BlahStructures
        else:
            if not directory:
                raise NameError('Must specify the directory of the relaxed structures!')
            try:
                base = read(os.path.join(directory,'_'.join(fragments),path,path+neb_name+'.traj'))
                base.old_energy=base.get_potential_energy()
            except:
                raise IOError('Relaxed trajectory file at %s does not exist!' % os.path.join(directory,'_'.join(fragments),path,path+neb_name+'.traj'))
            #Perform necessary translations
            if not translations==None:
                #Set non-indicated guys to translate 0.
                diff = len(base)-len(translations)
                translations = [0.]*diff + translations
                ads_size = len([a for a in base if a.tag==0])
                #Get vector between first and second metal atom (lattice value)
                trans = base.positions[-1*ads_size-1]-base.positions[-1*ads_size-2]
                #Translate in units of lattice value
                for i, b in enumerate(base):
                    b.position+=translations[i]*trans
        #add layers
        counter=0
        while counter<add_layers:
            base=self.add_layer(base)
            counter+=1
        #Build in ability to do single layer calcs. Messy that we start with two.....
        if add_layers==-1:
            base = base[2:]
        base.center(vacuum=vacuum,axis=(2))
        #SHAME SHAME!!!!
        #c = base.cell
        #c[0][0]+=.1
        #c[1][1]+=.2
        #base.set_cell(c)
        base.translate([.38364*vacuum,0.08574*vacuum,.18675*vacuum])        
        #No magnetizaton!!!
        base.set_initial_magnetic_moments(np.zeros(len(base)))
        #Interchange positions (to interchange indices) as needed
        if interchanges:
            npos = base.get_positions()
            npos[interchanges[0]]=npos[interchanges[1]]
            base.set_positions(npos)
        assert fixed_layers<= np.max(base.get_tags())
        #Fix the appropriate number of layers
        cons = FixAtoms(mask=[atom.tag>(np.max(base.get_tags())-fixed_layers) for atom in base])
        base.set_constraint(cons)
        
        return base

    #A function for adding layers to a given input structure. Will also adjust the unit cell. Not sure about constraints ATM.
    def add_layer(self,struc):
        pos = struc.get_positions()
        ntag = np.max(struc.get_tags())+1
        ocell = struc.cell
        del_z = pos[2][2]-pos[0][2]
        ncell = ocell+ np.array([[0,0,0],[0,0,0],[0,0,del_z]])    
        dist = struc.get_distance(0,1)
        #Deal with which of the 3 layers we need to add on....
        if np.abs(pos[0][0]-pos[2][0])==0.:
            n_y = np.abs(pos[0][1]+pos[2][1])/2.         
            n_x1=np.sqrt(3)*(np.abs(pos[0][1]-pos[2][1])/2.)+pos[0][0]
            n_x2=np.sqrt(3)*(np.abs(pos[0][1]-pos[2][1])/2.)+pos[1][0]
        elif pos[0][0]>pos[2][0]:
            n_y = pos[0][1]+(pos[0][1]-pos[2][1])
            n_x1=-1*np.sqrt(3)*(np.abs(pos[0][1]-pos[2][1])/2.)+pos[0][0]
            n_x2=-1*np.sqrt(3)*(np.abs(pos[0][1]-pos[2][1])/2.)+pos[1][0]
        else:
            n_y=pos[0][1]-2*(pos[0][1]-pos[2][1])
            n_x1,n_x2 =(pos[0][0],pos[1][0])
        nlayer = Atoms(struc[0].symbol+'2',[[n_x1,n_y,pos[0][2]-del_z],[n_x2,n_y,pos[1][2]-del_z]],tags=[ntag,ntag])
        nstruc = nlayer+struc
        nstruc.set_cell(ncell)
        return nstruc

