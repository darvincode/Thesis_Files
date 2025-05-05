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

os.chdir('/scratch_tmp/prj/ch_smiecs_epsrc/batch_scripts/Vacuum_Simulations/100M1T1BBNEW_01-03-2025_01_00_08/')

topology_file = ("interchange_gromacs.top")
coordinate_file = ("interchange_gromacs.gro")
structure = pmd.load_file(topology_file, xyz=coordinate_file)

# Export to OpenMM
system2 = structure.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=0.9*nanometers)

# Extract topology and positions
topology = structure.topology
positions = structure.positions

#Create directory for results
from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
dir_name = '100M1T1HOLDNEW_' + str(now)
os.makedirs(dir_name, exist_ok=True)

#Enter new directory
os.chdir("./" + dir_name)

time_step = 2 * openmm.unit.femtoseconds  # simulation timestep
temperature = 800 * openmm.unit.kelvin  # simulation temperature
friction = 1 / openmm.unit.picosecond  # friction constant

integrator = openmm.LangevinIntegrator(temperature, friction, time_step)

with open('/scratch_tmp/prj/ch_smiecs_epsrc/batch_scripts/Vacuum_Simulations//100M1T1BBNEW_01-03-2025_01_00_08/simulationm1_01-03-2025_03_48_15/system_NPT.xml') as input:
    system = openmm.XmlSerializer.deserialize(input.read())

simulation = Simulation(topology, system = system, integrator=integrator, state = '/scratch_tmp/prj/ch_smiecs_epsrc/batch_scripts/Vacuum_Simulations//100M1T1BBNEW_01-03-2025_01_00_08/simulationm1_01-03-2025_03_48_15/eq_NPT.state')
simulation.context.setParameter(MonteCarloBarostat.Temperature(), 800*openmm.unit.kelvin)
state = simulation.context.getState(getPositions=True)  # Get the current simulation state
box_vectors = state.getPeriodicBoxVectors()
simulation.context.reinitialize(True)
simulation.context.setPositions(positions)

integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
integrator.setConstraintTolerance(0.00001)


num_steps = 5000000  # number of steps to run

#importing the NVT information - equilibration (constant number of atoms volume and temperature )

#logging options
trj_freq = 100000  # number of steps per written trajectory frame
data_freq = 100000 # number of steps per written simulation statistics

# Add reporters
#pdb_reporter = openmm.app.PDBReporter("trajectory_equil_pdb.pdb", trj_freq)
dcd_reporter = openmm.app.DCDReporter("trajectory_equil_dcd.dcd", trj_freq)

state_data_reporter = openmm.app.StateDataReporter(
    "data_equil.csv",
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
#simulation.reporters.append(pdb_reporter)
simulation.reporters.append(dcd_reporter)
simulation.reporters.append(state_data_reporter)

# Run the simulation
simulation.step(num_steps) 

################################################################################################################################################################################################
# save the equilibration results to file
simulation.saveState('hold.state')
simulation.saveCheckpoint('hold.chk')

print('Performed 800K hold step' , 'temp = ', str(temperature), 'time = ', str((num_steps*2)/1000000), 'ns')

#Save system for reinitialization if needed
system = simulation.context.getSystem()
with open('hold.xml', 'w') as output:
    output.write(openmm.XmlSerializer.serialize(system))
