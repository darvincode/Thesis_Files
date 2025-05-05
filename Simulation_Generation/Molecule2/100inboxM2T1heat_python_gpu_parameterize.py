import rdkit
from rdkit import Chem
import os
import random
import pandas as pd
import time
import numpy as np
#import mdtraj as md
import nglview
import openmm
from openbabel import openbabel

from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.utils import get_data_file_path
from openff.toolkit.utils.toolkits import RDKitToolkitWrapper, OpenEyeToolkitWrapper, AmberToolsToolkitWrapper
from openff.units import unit
from pandas import read_csv

from openff.interchange import Interchange
from openff.interchange.components._packmol import UNIT_CUBE, pack_box, RHOMBIC_DODECAHEDRON
from openff.interchange.components._packmol import _max_dist_between_points, _compute_brick_from_box_vectors, _center_topology_at
from openmm.openmm import System
from openmm import MonteCarloBarostat
from openmm.app.simulation import Simulation

import subprocess
import parmed as pmd
from openmm.app import Simulation
from openmm import LangevinIntegrator
from openmm.unit import kelvin, picoseconds, nanometers
#from  rdkit  import  Chem
from  rdkit.Chem  import  rdDistGeom
from  rdkit.Chem.Draw  import  IPythonConsole
import warnings
warnings.filterwarnings('ignore')



dimer_openff = Molecule.from_file('darvin_mol_2_set_ang.mol')

#os.chdir('../')
#os.chdir('../outputs')

#Create directory for results
from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y_%H_%M_%S")

dir_name = '100M2T1BBNEW_' + str(now)
os.makedirs(dir_name, exist_ok=True)

#Enter new directory
os.chdir("./" + dir_name)

#Convert to OpenFF topology
openff_top = dimer_openff.to_topology()

number_of_chains=100
padding= 2 * unit.nanometer
box_shape= UNIT_CUBE
target_density= 1.0 * unit.gram / unit.milliliter

#### New code - topology + packmol coordinates
openff_top_100 = Topology.from_molecules([dimer_openff]*number_of_chains)

def get_coords_packmol(path_xyz):
    xyz = open(path_xyz)
    N = int(xyz.readline())
    header = xyz.readline()
    atom_symbol, coords = ([] for i in range (2))
    for line in xyz:
        atom,x,y,z = line.split()
        atom_symbol.append(atom)
        coords.append([float(x),float(y),float(z)])
    coords_arr = np.array(coords)
    return coords_arr
    
arr = get_coords_packmol('../molecule2_100packed120.xyz')

openff_top_100.set_positions(arr * unit.angstrom)
openff_top_100.box_vectors = (70 * unit.angstrom) * UNIT_CUBE #IMPORTANT this implements PBC


interchange = Interchange.from_smirnoff(topology=openff_top_100, 
                                        force_field=ForceField("openff-2.2.0.offxml"))

interchange.to_pdb("interchange.pdb")

interchange.to_gromacs('interchange_gromacs')

subprocess.run(['gmx', 'grompp', '-f', '../min.mdp', '-c', 'interchange_gromacs.gro', '-p', 'interchange_gromacs.top', '-o', 'interchange_gromacs.tpr', '-maxwarn', '10'], check=True) #performs GROMACs energy minimisation - rearrange atoms arrangement to lowest energy
subprocess.run(['gmx', 'mdrun', '-deffnm', 'interchange_gromacs'], check=True)

# Load GROMACS topology and coordinate files
topology_file = "interchange_gromacs.top"
coordinate_file = "interchange_gromacs.gro"
structure = pmd.load_file(topology_file, xyz=coordinate_file) #PARMED

# Export to OpenMM
system = structure.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=0.9*nanometers)

# Extract topology and positions
topology = structure.topology
positions = structure.positions
from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
dir_name = 'simulationm2_' + str(now)
os.makedirs(dir_name, exist_ok=True)

os.chdir("./" + dir_name)
# Define an integrator
integrator = LangevinIntegrator(298*kelvin, 1/picoseconds, 0.002*picoseconds)

# Set up the simulation
simulation = Simulation(topology, system, integrator)
simulation.context.setPositions(positions)

# Logging options.
trj_freq = 100  # number of steps per written trajectory frame
data_freq = 100  # number of steps per written simulation statistics

simulation.reporters.append(openmm.app.DCDReporter("trajectory_NVT_dcd_298eqramp_K.dcd", trj_freq))
assert simulation.system.usesPeriodicBoundaryConditions()
state = simulation.context.getState(getEnergy=True)
print(state.getPotentialEnergy())
simulation.minimizeEnergy()
print(state.getPotentialEnergy())

num_steps = 2000  # number of integration steps to run

# Integration options
time_step = 2 * openmm.unit.femtoseconds  # simulation timestep
temperature = 298 * openmm.unit.kelvin  # simulation temperature
friction = 1 / openmm.unit.picosecond  # friction constant



state_data_reporter = openmm.app.StateDataReporter(
    "data_NVT.csv",
    reportInterval=data_freq,
    step = True,             # writes the step number to each line
    time = True,             # writes the time (in ps)
    potentialEnergy = True,  # writes potential energy of the system (KJ/mole)
    kineticEnergy = True,    # writes the kinetic energy of the system (KJ/mole)
    totalEnergy = True,      # writes the total energy of the system (KJ/mole)
    temperature = True,      # writes the temperature (in K)
    volume = True,           # writes the volume (in nm^3)
    density = True)         # writes the density (in g/mL)

# Append state reporters
simulation.reporters.append(state_data_reporter)

#Simulation
print("Starting equilibration...")
start = time.process_time()

# Run the simulation
simulation.step(num_steps)

# save the equilibration results to file
simulation.saveState('eq.state')
simulation.saveCheckpoint('eq.chk')

end = time.process_time()
print(f"Elapsed time {end - start} seconds")

print('Performed NVT step' , 'temp = ', str(temperature), 'time = ', str((num_steps*2)/1000000), 'ns')

#Save system for reinitialization if needed
system = simulation.context.getSystem()
with open('equil.xml', 'w') as output:
    output.write(openmm.XmlSerializer.serialize(system))
