#!/usr/bin/env python3

#******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2019 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
#******************************************

# Script for the calculation of Wigner distributions from molden frequency files
#
# usage python wigner.py [-n <NUMBER>] <MOLDEN-FILE>

import copy
import math
import random
import sys
from optparse import OptionParser

np = True
import numpy

# some constants
CM_TO_HARTREE = 1./219474.6     #4.556335252e-6 # conversion factor from cm-1 to Hartree
HARTREE_TO_EV = 27.211396132    # conversion factor from Hartree to eV
U_TO_AMU = 1./5.4857990943e-4            # conversion from g/mol to amu
ANG_TO_BOHR = 1./0.529177211    #1.889725989      # conversion from Angstrom to bohr


NUMBERS = {'H':1, 'He':2,
'Li':3, 'Be':4, 'B':5, 'C':6,  'N':7,  'O':8, 'F':9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14,  'P':15,  'S':16, 'Cl':17, 'Ar':18,
'K':19, 'Ca':20,
'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,
'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
'Rb':37, 'Sr':38,
'Y':39,  'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48,
'In':49, 'Sn':50, 'Sb':51, 'Te':52,  'I':53, 'Xe':54,
'Cs':55, 'Ba':56,
'La':57, 
'Ce':58, 'Pr':59, 'Nd':60, 'Pm':61, 'Sm':62, 'Eu':63, 'Gd':64, 'Tb':65, 'Dy':66, 'Ho':67, 'Er':68, 'Tm':69, 'Yb':70, 'Lu':71,
         'Hf':72, 'Ta':73,  'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86, 
'Fr':87, 'Ra':88,
'Ac':89, 
'Th':90, 'Pa':91,  'U':92, 'Np':93, 'Pu':94, 'Am':95, 'Cm':96, 'Bk':97, 'Cf':98, 'Es':99,'Fm':100,'Md':101,'No':102,'Lr':103,
        'Rf':104,'Db':105,'Sg':106,'Bh':107,'Hs':108,'Mt':109,'Ds':110,'Rg':111,'Cn':112,
'Nh':113,'Fl':114,'Mc':115,'Lv':116,'Ts':117,'Og':118
}

# Atomic Weights of the most common isotopes
# From https://chemistry.sciences.ncsu.edu/msf/pdf/IsotopicMass_NaturalAbundance.pdf
MASSES = {'H' :   1.007825 * U_TO_AMU,
          'He':   4.002603 * U_TO_AMU,
          'Li':   7.016004 * U_TO_AMU,
          'Be':   9.012182 * U_TO_AMU,
          'B' :  11.009305 * U_TO_AMU,
          'C' :  12.000000 * U_TO_AMU,
          'N' :  14.003074 * U_TO_AMU,
          'O' :  15.994915 * U_TO_AMU,
          'F' :  18.998403 * U_TO_AMU,
          'Ne':  19.992440 * U_TO_AMU,
          'Na':  22.989770 * U_TO_AMU,
          'Mg':  23.985042 * U_TO_AMU,
          'Al':  26.981538 * U_TO_AMU,
          'Si':  27.976927 * U_TO_AMU,
          'P' :  30.973762 * U_TO_AMU,
          'S' :  31.972071 * U_TO_AMU,
          'Cl':  34.968853 * U_TO_AMU,
          'Ar':  39.962383 * U_TO_AMU,
          'K' :  38.963707 * U_TO_AMU,
          'Ca':  39.962591 * U_TO_AMU,
          'Sc':  44.955910 * U_TO_AMU,
          'Ti':  47.947947 * U_TO_AMU,
          'V' :  50.943964 * U_TO_AMU,
          'Cr':  51.940512 * U_TO_AMU,
          'Mn':  54.938050 * U_TO_AMU,
          'Fe':  55.934942 * U_TO_AMU,
          'Co':  58.933200 * U_TO_AMU,
          'Ni':  57.935348 * U_TO_AMU,
          'Cu':  62.929601 * U_TO_AMU,
          'Zn':  63.929147 * U_TO_AMU,
          'Ga':  68.925581 * U_TO_AMU,
          'Ge':  73.921178 * U_TO_AMU,
          'As':  74.921596 * U_TO_AMU,
          'Se':  79.916522 * U_TO_AMU,
          'Br':  78.918338 * U_TO_AMU,
          'Kr':  83.911507 * U_TO_AMU,
          'Rb':  84.911789 * U_TO_AMU,
          'Sr':  87.905614 * U_TO_AMU,
          'Y' :  88.905848 * U_TO_AMU,
          'Zr':  89.904704 * U_TO_AMU,
          'Nb':  92.906378 * U_TO_AMU,
          'Mo':  97.905408 * U_TO_AMU,
          'Tc':  98.907216 * U_TO_AMU,
          'Ru': 101.904350 * U_TO_AMU,
          'Rh': 102.905504 * U_TO_AMU,
          'Pd': 105.903483 * U_TO_AMU,
          'Ag': 106.905093 * U_TO_AMU,
          'Cd': 113.903358 * U_TO_AMU,
          'In': 114.903878 * U_TO_AMU,
          'Sn': 119.902197 * U_TO_AMU,
          'Sb': 120.903818 * U_TO_AMU,
          'Te': 129.906223 * U_TO_AMU,
          'I' : 126.904468 * U_TO_AMU,
          'Xe': 131.904154 * U_TO_AMU,
          'Cs': 132.905447 * U_TO_AMU,
          'Ba': 137.905241 * U_TO_AMU,
          'La': 138.906348 * U_TO_AMU,
          'Ce': 139.905435 * U_TO_AMU,
          'Pr': 140.907648 * U_TO_AMU,
          'Nd': 141.907719 * U_TO_AMU,
          'Pm': 144.912744 * U_TO_AMU,
          'Sm': 151.919729 * U_TO_AMU,
          'Eu': 152.921227 * U_TO_AMU,
          'Gd': 157.924101 * U_TO_AMU,
          'Tb': 158.925343 * U_TO_AMU,
          'Dy': 163.929171 * U_TO_AMU,
          'Ho': 164.930319 * U_TO_AMU,
          'Er': 165.930290 * U_TO_AMU,
          'Tm': 168.934211 * U_TO_AMU,
          'Yb': 173.938858 * U_TO_AMU,
          'Lu': 174.940768 * U_TO_AMU,
          'Hf': 179.946549 * U_TO_AMU,
          'Ta': 180.947996 * U_TO_AMU,
          'W' : 183.950933 * U_TO_AMU,
          'Re': 186.955751 * U_TO_AMU,
          'Os': 191.961479 * U_TO_AMU,
          'Ir': 192.962924 * U_TO_AMU,
          'Pt': 194.964774 * U_TO_AMU,
          'Au': 196.966552 * U_TO_AMU,
          'Hg': 201.970626 * U_TO_AMU,
          'Tl': 204.974412 * U_TO_AMU,
          'Pb': 207.976636 * U_TO_AMU,
          'Bi': 208.980383 * U_TO_AMU,
          'Po': 208.982416 * U_TO_AMU,
          'At': 209.987131 * U_TO_AMU,
          'Rn': 222.017570 * U_TO_AMU,
          'Fr': 223.019731 * U_TO_AMU, 
          'Ra': 226.025403 * U_TO_AMU,
          'Ac': 227.027747 * U_TO_AMU, 
          'Th': 232.038050 * U_TO_AMU, 
          'Pa': 231.035879 * U_TO_AMU, 
          'U' : 238.050783 * U_TO_AMU, 
          'Np': 237.048167 * U_TO_AMU,
          'Pu': 244.064198 * U_TO_AMU, 
          'Am': 243.061373 * U_TO_AMU, 
          'Cm': 247.070347 * U_TO_AMU, 
          'Bk': 247.070299 * U_TO_AMU, 
          'Cf': 251.079580 * U_TO_AMU, 
          'Es': 252.082972 * U_TO_AMU,
          'Fm': 257.095099 * U_TO_AMU,
          'Md': 258.098425 * U_TO_AMU,
          'No': 259.101024 * U_TO_AMU,
          'Lr': 262.109692 * U_TO_AMU,
          'Rf': 267. * U_TO_AMU,
          'Db': 268. * U_TO_AMU,
          'Sg': 269. * U_TO_AMU,
          'Bh': 270. * U_TO_AMU,
          'Hs': 270. * U_TO_AMU,
          'Mt': 278. * U_TO_AMU,
          'Ds': 281. * U_TO_AMU,
          'Rg': 282. * U_TO_AMU,
          'Cn': 285. * U_TO_AMU,
          'Nh': 286. * U_TO_AMU,
          'Fl': 289. * U_TO_AMU,
          'Mc': 290. * U_TO_AMU,
          'Lv': 293. * U_TO_AMU,
          'Ts': 294. * U_TO_AMU,
          'Og': 294. * U_TO_AMU
}

# Isotopes used for the masses
ISOTOPES={'H' : 'H-1' ,
          'He': 'He-4',
          'Li': 'Li-7',
          'Be': 'Be-9*',
          'B' : 'B_11' ,
          'C' : 'C-12' ,
          'N' : 'N-14' ,
          'O' : 'O-16' ,
          'F' : 'F-19*' ,
          'Ne': 'Ne-20',
          'Na': 'Na-23*',
          'Mg': 'Mg-24',
          'Al': 'Al-27*',
          'Si': 'Si-28',
          'P' : 'P-31*' ,
          'S' : 'S-32' ,
          'Cl': 'Cl-35',
          'Ar': 'Ar-40',
          'K' : 'K-39' ,
          'Ca': 'Ca-40',
          'Sc': 'Sc-45*',
          'Ti': 'Ti-48',
          'V' : 'V-51' ,
          'Cr': 'Cr-52',
          'Mn': 'Mn-55*',
          'Fe': 'Fe-56',
          'Co': 'Co-59*',
          'Ni': 'Ni-58',
          'Cu': 'Cu-63',
          'Zn': 'Zn-64',
          'Ga': 'Ga-69',
          'Ge': 'Ge-74',
          'As': 'As-75*',
          'Se': 'Se-80',
          'Br': 'Br-79',
          'Kr': 'Kr-84',
          'Rb': 'Rb-85',
          'Sr': 'Sr-88',
          'Y' : 'Y-89*' ,
          'Zr': 'Zr-90',
          'Nb': 'Nb-93*',
          'Mo': 'Mo-98',
          'Tc': 'Tc-98',
          'Ru': 'Ru-102',
          'Rh': 'Rh-103*',
          'Pd': 'Pd-106',
          'Ag': 'Ag-107',
          'Cd': 'Cd-114',
          'In': 'In-115',
          'Sn': 'Sn-120',
          'Sb': 'Sb-121',
          'Te': 'Te-130',
          'I' : 'I-127*' ,
          'Xe': 'Xe-132',
          'Cs': 'Cs-133*',
          'Ba': 'Ba-138',
          'La': 'La-139',
          'Ce': 'Ce-140',
          'Pr': 'Pr-141*',
          'Nd': 'Nd-142',
          'Pm': 'Pm-145',
          'Sm': 'Sm-152',
          'Eu': 'Eu-153',
          'Gd': 'Gd-158',
          'Tb': 'Tb-159*',
          'Dy': 'Dy-164',
          'Ho': 'Ho-165*',
          'Er': 'Er-166',
          'Tm': 'Tm-169*',
          'Yb': 'Yb-174',
          'Lu': 'Lu-175',
          'Hf': 'Hf-180',
          'Ta': 'Ta-181',
          'W' : 'W-184' ,
          'Re': 'Re-187',
          'Os': 'Os-192',
          'Ir': 'Ir-193',
          'Pt': 'Pt-195',
          'Au': 'Au-197*',
          'Hg': 'Hg-202',
          'Tl': 'Tl-205',
          'Pb': 'Pb-208',
          'Bi': 'Bi-209*',
          'Po': 'Po-209',
          'At': 'At-210',
          'Rn': 'Rn-222',
          'Fr': 'Fr-223', 
          'Ra': 'Ra-226',
          'Ac': 'Ac-227', 
          'Th': 'Th-232*', 
          'Pa': 'Pa-231*', 
          'U' : 'U-238' , 
          'Np': 'Np-237',
          'Pu': 'Pu-244', 
          'Am': 'Am-243', 
          'Cm': 'Cm-247', 
          'Bk': 'Bk-247', 
          'Cf': 'Cf-251', 
          'Es': 'Es-252',
          'Fm': 'Fm-257',
          'Md': 'Md-258',
          'No': 'No-259',
          'Lr': 'Lr-262',
              'Rf': 'Rf-267',
              'Db': 'Db-268',
              'Sg': 'Sg-269',
              'Bh': 'Bh-270',
              'Hs': 'Hs-270',
              'Mt': 'Mt-278',
              'Ds': 'Ds-281',
              'Rg': 'Rg-282',
              'Cn': 'Cn-285',
              'Nh': 'Nh-286',
              'Fl': 'Fl-289',
              'Mc': 'Mc-290',
              'Lv': 'Lv-293',
              'Ts': 'Ts-294',
              'Og': 'Og-294'
}



# thresholds
LOW_FREQ = 10.0 # threshold in cm^-1 for ignoring rotational and translational low frequencies
KTR = False

class ATOM:
  def __init__(self,symb='??',num=0.,coord=[0.,0.,0.],m=0.,veloc=[0.,0.,0.]):
    self.symb  = symb
    self.num   = num
    self.coord = coord
    self.mass  = m
    self.veloc = veloc
    self.Ekin=0.5*self.mass * sum( [ self.veloc[i]**2 for i in range(3) ] )

  def __str__(self):
    s ='%2s % 5.1f '               % (self.symb, self.num)
    s+='% 12.8f % 12.8f % 12.8f '  % tuple(self.coord)
    s+='% 12.8f '                  % (self.mass/U_TO_AMU)
    s+='% 12.8f % 12.8f % 12.8f'   % tuple(self.veloc)
    return s

class INITCOND:
  def __init__(self,atomlist=[],eref=0.,epot_harm=0.):
    self.atomlist=atomlist
    self.eref=eref
    self.Epot_harm=epot_harm
    self.natom=len(atomlist)
    self.Ekin=sum( [atom.Ekin for atom in self.atomlist] )
    self.nstate=0
    self.Epot=epot_harm

  def __str__(self):
    s='Atoms\n'
    for atom in self.atomlist:
      s+=str(atom)+'\n'
    s+='Ekin      % 16.12f a.u.\n' % (self.Ekin)
    s+='Epot_harm % 16.12f a.u.\n' % (self.Epot_harm)
    s+='Epot      % 16.12f a.u.\n' % (self.Epot)
    s+='Etot_harm % 16.12f a.u.\n' % (self.Epot_harm+self.Ekin)
    s+='Etot      % 16.12f a.u.\n' % (self.Epot+self.Ekin)
    s+='\n\n'
    return s

# TODO: remove this
def get_mass(symb):
    try:
      return MASSES[symb]
    except KeyError:
      print('No default mass for atom %s' % (symb))
      sys.exit(1)


# TODO: Remove this
def import_from_molden(filename, scaling, flag, unique_atom_names):
  '''This function imports atomic coordinates and normal modes from a MOLDEN
file. Returns molecule and modes as the other function does.
'''
  f=open(filename)
  data=f.readlines()
  f.close()

  # find coordinate block
  iline=0
  while not 'FR-COORD' in data[iline]:
    iline+=1
    if iline==len(data):
      print('Could not find coordinates in %s!' % (filename))
      sys.exit(1)
  # get atoms
  iline+=1
  natom=0
  molecule=[]
  while not '[' in data[iline]:
    f=data[iline].split()
    symb=f[0].lower().title()
    num=NUMBERS[symb]
    coord=[ float(f[i+1]) for i in range(3) ]
    natom+=1
    mass=get_mass(symb)
    unique_atom_names.add(symb)
    molecule.append(ATOM(symb,num,coord,mass))
    iline+=1

  # find number of frequencies
  iline=-1
  nmodes=-1
  while True:
    iline+=1
    if iline==len(data):
      nmodes=3*natom
      break
    line=data[iline]
    if 'N_FREQ' in line:
      nmodes=int(data[iline+1])
      break

  # warn, if too few normal modes were found
  if nmodes<3*natom:
    print('*'*51+'\nWARNING: Less than 3*N_atom normal modes extracted!\n'+'*'*51+'\n')

  # obtain all frequencies, including low ones
  iline=0
  modes=[]
  while not '[FREQ]' in data[iline]:
    iline+=1
  iline+=1
  for imode in range(nmodes):
    try:
      mode={'freq':float(data[iline+imode])*CM_TO_HARTREE * scaling}
      modes.append(mode)
    except ValueError:
      print('*'*51+'\nWARNING: Less than 3*N_atom normal modes, but no [N_FREQ] keyword!\n'+'*'*51+'\n')
      nmodes=imode
      break

  # obtain normal coordinates
  iline=0
  while not 'FR-NORM-COORD' in data[iline]:
    iline+=1
  iline+=1
  for imode in range(nmodes):
    iline+=1
    move=[]
    for iatom in range(natom):
      f=data[iline].split()
      move.append([ float(f[i]) for i in range(3) ])
      iline+=1
    modes[imode]['move']=move
    # normalization stuff
    norm = 0.0
    for j, atom in enumerate(molecule):
      for xyz in range(3):
        norm += modes[imode]['move'][j][xyz]**2
    norm = math.sqrt(norm)
    if norm ==0.0 and modes[imode]['freq']>=LOW_FREQ*CM_TO_HARTREE:
      print('WARNING: Displacement vector of mode %i is null vector. Ignoring this mode!' % (imode+1))
      modes[imode]['freq']=0.


  newmodes=[]
  for imode in range(nmodes):
    if modes[imode]['freq']<0.:
      print('Detected negative frequency!')
      sys.exit(1)
    if modes[imode]['freq']>=LOW_FREQ*CM_TO_HARTREE:
      newmodes.append(modes[imode])
  modes=newmodes  

  nmodes = len(modes)
  modes = convert_orca_normal_modes(modes, molecule)

  return molecule, modes

def wigner(Q, P, mode):
    """This function calculates the Wigner distribution for
a single one-dimensional harmonic oscillator.
Q contains the dimensionless coordinate of the
oscillator and P contains the corresponding momentum.
n is the number of the vibrational state (default 0).
The function returns a probability for this set of parameters."""
    n = 0
    return (math.exp(-Q**2) * math.exp(-P**2), 0.)

def get_center_of_mass(molecule):
    """This function returns a list containing the center of mass
of a molecule."""
    mass = 0.0
    for atom in molecule:
        mass += atom.mass
    com = [0.0 for xyz in range(3)]
    for atom in molecule:
        for xyz in range(3):
            com[xyz] += atom.coord[xyz] * atom.mass / mass
    return com

def restore_center_of_mass(molecule, ic):
    """This function restores the center of mass for the distorted
geometry of an initial condition."""
    # calculate original center of mass
    com = get_center_of_mass(molecule)
    # caluclate center of mass for initial condition of molecule
    com_distorted = get_center_of_mass(ic)
    # get difference vector and restore original center of mass
    diff = [com[xyz] - com_distorted[xyz] for xyz in range(3)]
    for atom in ic:
        for xyz in range(3):
            atom.coord[xyz] += diff[xyz]


def convert_orca_normal_modes(modes, molecule):
  #apply transformations to normal modes
  converted_modes = copy.deepcopy(modes)
  for imode in range(len(modes)):
    norm = 0.0
    for j, atom in enumerate(molecule):
      for xyz in range(3):
        norm += modes[imode]['move'][j][xyz]**2*atom.mass/U_TO_AMU
    norm = math.sqrt(norm)
    if norm == 0.0 and modes[imode]['freq']>=LOW_FREQ*CM_TO_HARTREE:
      print('WARNING: Displacement vector of mode %i is null vector. Ignoring this mode!' % (imode+1))
      # This is not expected, let's stop
      sys.exit(1)
      #for normmodes in allmodes:
      #  normmodes[imode]['freq']=0.0

    for j, atom in enumerate(molecule):
      for xyz in range(3):
        converted_modes[imode]['move'][j][xyz] /= norm/math.sqrt(atom.mass/U_TO_AMU)

  return converted_modes


def sample_initial_condition(molecule, modes):
  """This function samples a single initial condition from the
modes and atomic coordinates by the use of a Wigner distribution.
The first atomic dictionary in the molecule list contains also
additional information like kinetic energy and total harmonic
energy of the sampled initial condition.
Method is based on L. Sun, W. L. Hase J. Chem. Phys. 133, 044313
(2010) nonfixed energy, independent mode sampling."""
  # copy the molecule in equilibrium geometry
  atomlist = copy.deepcopy(molecule) # initialising initial condition object
  Epot = 0.0
  for mode in modes: # for each uncoupled harmonatomlist oscillator
    while True:
      # get random Q and P in the interval [-5,+5]
      # this interval is good for vibrational ground state
      # should be increased for higher states
      # TODO: needs to be restructured: first obtain temperature, then draw random numbers, then compute wigner probability
      random_Q = random.random()*10.0 - 5.0
      random_P = random.random()*10.0 - 5.0
      # calculate probability for this set of P and Q with Wigner distr.
      probability = wigner(random_Q, random_P, mode)
      if probability[0]>1. or probability[0]<0.:
        print('WARNING: wrong probability %f detected!' % (probability[0]))
        sys.exit(1)
      elif probability[0] > random.random():
        break # coordinates accepted
    # now transform the dimensionless coordinate into a real one
    # paper says, that freq_factor is sqrt(2*PI*freq)
    # QM programs directly give angular frequency (2*PI is not needed)
    freq_factor = math.sqrt(mode['freq'])
    # Higher frequencies give lower displacements and higher momentum.
    # Therefore scale random_Q and random_P accordingly:
    random_Q /= freq_factor
    random_P *= freq_factor
    # add potential energy of this mode to total potential energy
    Epot += 0.5 * mode['freq']**2 * random_Q**2
    for i, atom in enumerate(atomlist): # for each atom
      for xyz in range(3): # and each direction
        # distort geometry according to normal mode movement
        # and unweigh mass-weighted normal modes
        atom.coord[xyz] += random_Q * mode['move'][i][xyz] * math.sqrt(1./atom.mass)

  if not KTR:
    restore_center_of_mass(molecule, atomlist)

  ic = INITCOND(atomlist, 0., Epot)
  return ic


def create_initial_conditions_list(amount, molecule, modes):
    """This function creates 'amount' initial conditions from the
data given in 'molecule' and 'modes'. Output is returned
as a list containing all initial condition objects."""
    print('Sampling initial conditions')
    ic_list = []
    for i in range(amount):
        # sample the initial condition
        ic = sample_initial_condition(molecule, modes)
        ic_list.append(ic)
    return ic_list


def make_dyn_file(ic_list,filename):
  fl=open(filename,'w')
  string=''
  for i,ic in enumerate(ic_list):
    string+='%i\n%i\n' % (ic.natom,i)
    for atom in ic.atomlist:
      string+='%s' % (atom.symb)
      for j in range(3):
        string+=' %f' % (atom.coord[j]/ANG_TO_BOHR)
      string+='\n'
  fl.write(string)
  fl.close()


# TODO: Create a safe wrapper
class Wigner():

    def __init__(self, molecule, modes, seed):
        # TODO: Initialize internal structures
        # We need to figure out what the format of inputs should be here
        # and do the conversion.
        self.molecule = molecule
        self.modes = convert_orca_normal_modes(modes, molecule)
        random.seed(seed)

    def set_random_seed(self, seed):
        # TODO: Use our own instance of random number generator
        self.rnd = random.Random(seed)
        rnd.seed(seed)

    def get_sample(self): # remove_com -> Bool
        # TODO: Convert to a format that AiiDa can understand
        # TODO: Remove COM here, based on input parameter
        return sample_initial_condition(self.molecule, self.modes)


def main():
  '''Main routine'''

  usage='''
Wigner.py [options] filename.molden

This script reads a MOLDEN file containing frequencies and normal modes [1]
and generates a Wigner distribution of geometries and velocities.

The creation of the geometries and velocities is based on the
sampling of the Wigner distribution of a quantum harmonic oscillator,
as described in [2] (non-fixed energy, independent mode sampling).

[1] http://www.cmbi.ru.nl/molden/molden_format.html
[2] L. Sun, W. L. Hase: J. Chem. Phys. 133, 044313 (2010)
'''

  description=''

  parser = OptionParser(usage=usage, description=description)
  parser.add_option('-n', dest='n', type=int, nargs=1, default=3, help="Number of geometries to be generated (integer, default=3)")
  parser.add_option('-s', dest='s', type=float, nargs=1, default=1.0, help="Scaling factor for the energies (float, default=1.0)")
  parser.add_option('-L', dest='L', type=float, nargs=1, default=10.0, help="Discard frequencies below this value in cm-1 (float, default=10.)")
  parser.add_option('-o', dest='outfile', type=str, nargs=1,
        default='initconds_new.xyz', help="Output filename")
  parser.add_option('-x', dest='X', action='store_true',help="Generate a xyz file with the sampled geometries in addition to the initconds file")

  parser.add_option('-r', dest='r', type=int, nargs=1, default=16661, help="Seed for the random number generator (integer, default=16661)")
  parser.add_option('-f', dest='f', type=int, nargs=1, default='0', help="Define the type of read normal modes. 0 for automatic assignement, 1 for gaussian-type normal modes (Gaussian, Turbomole, Q-Chem, ADF, Orca), 2 for cartesian normal modes (Molcas, Molpro), 3 for Columbus-type (Columbus), or 4 for mass-weighted. (integer, default=0)")
  parser.add_option('--keep_trans_rot', dest='KTR', action='store_true',help="Keep translational and rotational components")
  
  (options, args) = parser.parse_args()

  random.seed(options.r)
  amount=options.n
  if len(args)==0:
    print(usage)
    sys.exit(1)
  filename=args[0]
  outfile=options.outfile
  scaling=options.s
  flag=options.f
  global LOW_FREQ
  LOW_FREQ=max(0.0000001,options.L)

  print('''Initial condition generation started...
INPUT  file                  = "%s"
OUTPUT file                  = "%s"
Number of geometries         = %i
Random number generator seed = %i''' % (filename, outfile, options.n, options.r))
  if scaling!=1.0:
    print('Scaling factor               = %f\n' % (scaling))

  global KTR
  KTR=options.KTR

  unique_atom_names = set()

  molecule, modes = import_from_molden(filename, scaling, flag,
          unique_atom_names)

  string = 'Assumed Isotopes: '
  for i in unique_atom_names:
    string += ISOTOPES[i] + ' '
  string+='\nIsotopes with * are pure isotopes.\n'
  print(string)

  string = 'Frequencies (cm^-1) used in the calculation:\n'
  for i,mode in enumerate(modes):
    string+='%4i %12.4f\n' % (i+1,mode['freq']/CM_TO_HARTREE)
  print(string)

  ic_list = create_initial_conditions_list(amount, molecule, modes)

  make_dyn_file(ic_list, outfile)

if __name__ == '__main__':
    main()
