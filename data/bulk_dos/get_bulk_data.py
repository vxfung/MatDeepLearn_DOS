from pymatgen import MPRester
from pymatgen.io.cif import CifWriter
import csv
import sys
import os
import pickle 
import numpy as np
from pymatgen.electronic_structure.core import Orbital, Spin

print('start')
###get API key from materials project login dashboard online
API_KEY='XXX'
mpr = MPRester(API_KEY)

download=True
if download==True:
    data = mpr.query(criteria={}, properties=["task_id", "formation_energy_per_atom"])
    filename = 'data_dump.pkl'
    outfile = open(filename,'wb')
    pickle.dump(data, outfile)
    outfile.close()
else:
    infile = open("data_dump.pkl",'rb')
    data = pickle.load(infile)
    infile.close()

print(len(data), data[0])