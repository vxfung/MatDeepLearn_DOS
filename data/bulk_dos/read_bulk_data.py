from pymatgen import MPRester
from pymatgen.io.cif import CifWriter
import csv
import sys
import os
import ase
from ase import io, Atoms
from ase.io import vasp
from ase.db import connect
from pymatgen.io.ase import AseAtomsAdaptor
import pickle
import numpy as np

print('start')   
###get API key from materials project login dashboard online
API_KEY='XXX'
mpr = MPRester(API_KEY)

filename = 'data_dump.pkl'
infile  = open(filename,'rb')
data = pickle.load(infile)
infile.close()

if not os.path.exists('data'):
    os.mkdir('data')   
    
adaptor = AseAtomsAdaptor()
count=0
lengths=[]

count=0
for z in range(0, len(data)):
    
    energy=data[z]["formation_energy_per_atom"]
    if energy > -500 and energy < 100:        
        try:
          out_temp=mpr.get_dos_by_material_id(data[z]["task_id"])
        except Exception as e:
          print(e)     
               
        if out_temp != None:
        
          dos_temp=out_temp.get_site_spd_dos(out_temp.structure[0])    
          orb = list(dos_temp.keys())
          fermi = dos_temp[orb[0]].efermi
          length=dos_temp[list(dos_temp.keys())[0]].get_densities(spin=None).shape
          dos=np.zeros((len(out_temp.structure), 4, length[0]))   
          for i in range(0, len(out_temp.structure)):  
            dos_temp=out_temp.get_site_spd_dos(out_temp.structure[i])
            shape=dos_temp[orb[0]].get_densities(spin=None).shape
            ##sum over orbitals
            for j in range(0, 4):
              if j == 0:
                dos[i,j,:]=dos_temp[orb[j]].energies - fermi
              else:
                dos[i,j,:]=dos_temp[orb[j-1]].get_densities(spin=None)    
          
          ##write structure and dos                
          ase_structure = adaptor.get_atoms(out_temp.structure)      
          if length[0] == 2001:   
            np.save("data/" +str(data[z]["task_id"])+'.npy', dos)            
            ase.io.vasp.write_vasp("data/" + str(data[z]["task_id"])+".vasp", ase_structure, vasp5=True)        
            ##placeholder  
            with open("data/" + 'targets.csv', 'a') as f:
              f.write(str(data[z]["task_id"])+','+str(energy)+','+str('0.00') + '\n')  
    
    count = count + 1
    print(count)
