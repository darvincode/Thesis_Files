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

os.chdir('100M1T1BBNEW_26-02-2025_10_57_57') #########################################

# Load GROMACS topology and coordinate files
topology_file = "interchange_gromacs.top"
coordinate_file = "interchange_gromacs.gro"
structure = pmd.load_file(topology_file, xyz=coordinate_file)
os.chdir("./" + 'simulation_mono_26-02-2025_13_27_54') ####################################
with open('equil.xml') as input:
    system = openmm.XmlSerializer.deserialize(input.read())
sys = structure.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=0.9*nanometers)
top = structure.topology
positions = structure.positions
from datetime import datetime
# Define an integrator
integrator = LangevinIntegrator(298*kelvin, 1/picoseconds, 0.002*picoseconds)

# Set up the simulation
simulation = Simulation(top, system, integrator, state='eq.state')
simulation.context.setPositions(positions)


##HEATING RAMP #NPT from Interchange run the simulation
trj_freq = 10000  # number of steps per written trajectory frame
data_freq = 10000 # number of steps per written simulation statistics
time_step = 1 * openmm.unit.femtoseconds  # simulation timestep
temperature = 298 * openmm.unit.kelvin  # simulation temperature
friction = 1 / openmm.unit.picosecond  # friction constant

integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
state = simulation.context.getState(getPositions=True, getVelocities=True)
positions = state.getPositions()
velocities = state.getVelocities()
box_vectors = state.getPeriodicBoxVectors()
#Add barostat for NPT ensemble
barostat = system.addForce(MonteCarloBarostat(1, 298))

dcd_reporter = openmm.app.DCDReporter(f"trajectory_NPT_dcd_298eqramp_K.dcd", trj_freq)
state_data_reporter = openmm.app.StateDataReporter(
        f"data_NPT_298eqrampK.csv",
        reportInterval=data_freq,
        step=True,             # writes the step number to each line
        time=True,             # writes the time (in ps)
        potentialEnergy=True,  # writes potential energy of the system (KJ/mole)
        kineticEnergy=True,    # writes the kinetic energy of the system (KJ/mole)
        totalEnergy=True,      # writes the total energy of the system (KJ/mole)
        temperature=True,      # writes the temperature (in K)
        volume=True,           # writes the volume (in nm^3)
        density=True)          # writes the density (in g/mL)


checkpoint_reporter = openmm.app.checkpointreporter.CheckpointReporter(f"checkpoint_prod_tempramp298eqrampK.chk", trj_freq, writeState=True)
simulation = Simulation(top, system, integrator)
simulation.reporters.append(dcd_reporter)
simulation.reporters.append(state_data_reporter)
simulation.reporters.append(checkpoint_reporter)


integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
simulation.context.setPositions(positions)
simulation.context.setVelocities(velocities)
num_steps = 1000000  # number of integration steps to run hold at 298K
print('Running simulation. Temp =', simulation.context.getIntegrator().getTemperature(), 'pressure = ', simulation.context.getParameter(MonteCarloBarostat.Pressure()) , 'Time = ',(num_steps*1)/1000000, 'ns')
simulation.step(num_steps)
print('Ran simulation. Temp =', temperature, 'pressure =', simulation.context.getParameter(MonteCarloBarostat.Pressure()) , 'Time = ',(num_steps*1)/1000000, 'ns')

start_temp = 298
end_temp = 800
quench_rate = 1 #K/ps
heat_ramp = np.arange(start_temp, end_temp + quench_rate, quench_rate).tolist()
heat_ramp #.reverse()

for i in heat_ramp: 
    num_steps = 1000 # steps that equals 1 ps (because we are working in K/ps)
    simulation.reporters.clear()
    temperature = i*openmm.unit.kelvin
    simulation.context.setParameter(MonteCarloBarostat.Temperature(), temperature)
    integrator.setTemperature(temperature)
    # Create new reporters with temperature in the title
    dcd_reporter = openmm.app.DCDReporter(f"trajectory_NPT_dcd_heatramp_{str(i).replace('.', '-')}K.dcd", 100)
    state_data_reporter = openmm.app.StateDataReporter(
        f"data_NPT_tempramp_{str(i).replace('.', '-')}K.csv",
        reportInterval=100,
        step=True,             # writes the step number to each line
        time=True,             # writes the time (in ps)
        potentialEnergy=True,  # writes potential energy of the system (KJ/mole)
        kineticEnergy=True,    # writes the kinetic energy of the system (KJ/mole)
        totalEnergy=True,      # writes the total energy of the system (KJ/mole)
        temperature=True,      # writes the temperature (in K)
        volume=True,           # writes the volume (in nm^3)
        density=True)          # writes the density (in g/mL)
    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(state_data_reporter)
    simulation.step(num_steps)
    print('Ran simulation. Temp =', temperature, 'pressure =', simulation.context.getParameter(MonteCarloBarostat.Pressure()) , 'Time = ',(num_steps*1)/1000000, 'ns')

simulation.saveState('eq_NPT.state')
simulation.saveCheckpoint('eq_NPT.chk')

#Save system for reinitialization if needed
system = simulation.context.getSystem()
with open('system_NPT.xml', 'w') as output:
    output.write(openmm.XmlSerializer.serialize(system))

 
################################################################################################################################################################################################

simulation.loadCheckpoint('eq_NPT.chk')
eq_state = simulation.context.getState(getVelocities=True, getPositions=True)
positions = eq_state.getPositions()
velocities = eq_state.getVelocities()

integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
integrator.setConstraintTolerance(0.00001)

top = structure.topology
system = simulation.context.getSystem()
simulation.context.reinitialize(True)

simulation2 = Simulation(simulation.topology, system, integrator)
simulation2.context.setPositions(positions)
simulation2.context.setVelocities(velocities)

num_steps = 15000000  # number of integration steps to run

# Logging options.
trj_freq = 100000  # number of steps per written trajectory frame
data_freq = 100000 # number of steps per written simulation statistics

# Integration options
time_step = 3 * openmm.unit.femtoseconds  # simulation timestep
temperature = 300 * openmm.unit.kelvin  # simulation temperature
friction = 1 / openmm.unit.picosecond  # friction constant

#Add reporters

#pdb_reporter = openmm.app.PDBReporter("trajectory_prod_pdb.pdb", num_steps)
dcd_reporter = openmm.app.DCDReporter("trajectory_prodhold_dcd.dcd", trj_freq)
checkpoint_reporter = openmm.app.checkpointreporter.CheckpointReporter("hold.chk", 500, writeState=True)

state_data_reporter = openmm.app.StateDataReporter(
    "datahold_prod.csv",
    reportInterval=data_freq,
    step = True,             # writes the step number to each line
    time = True,             # writes the time (in ps)
    potentialEnergy = True,  # writes potential energy of the system (KJ/mole)
    kineticEnergy = True,    # writes the kinetic energy of the system (KJ/mole)
    totalEnergy = True,      # writes the total energy of the system (KJ/mole)
    temperature = True,      # writes the temperature (in K)
    volume = True,           # writes the volume (in nm^3)
    density = True)         # writes the density (in g/mL)

simulation2.reporters.append(dcd_reporter)
simulation2.reporters.append(checkpoint_reporter)
simulation2.reporters.append(state_data_reporter)
# Run the simulation
simulation2.step(num_steps)

################################################################################################################################################################################################


simulation.loadCheckpoint('hold.chk')
eq_state = simulation.context.getState(getVelocities=True, getPositions=True)
positions = eq_state.getPositions()
velocities = eq_state.getVelocities()

integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
integrator.setConstraintTolerance(0.00001)

top = structure.topology
system = simulation.context.getSystem()
simulation.context.reinitialize(True)

simulation2 = Simulation(simulation.topology, system, integrator)
simulation2.context.setPositions(positions)
simulation2.context.setVelocities(velocities)

start_temp = 800
end_temp = 298
quench_rate = 1 #K/ps
heat_ramp = np.arange(start_temp, end_temp + quench_rate, quench_rate).tolist()
heat_ramp.reverse()

for i in heat_ramp: 
    num_steps = 10000 # steps that equals 10ns (because we are working in K/ps)
    simulation.reporters.clear()
    temperature = i*openmm.unit.kelvin
    simulation.context.setParameter(MonteCarloBarostat.Temperature(), temperature)
    integrator.setTemperature(temperature)
    # Create new reporters with temperature in the title
    dcd_reporter = openmm.app.DCDReporter(f"trajectory_NPT_dcd_coolramp_{str(i).replace('.', '-')}K.dcd", 100)
    state_data_reporter = openmm.app.StateDataReporter(
        f"data_NPT_tempramp_{str(i).replace('.', '-')}K.csv",
        reportInterval=100,
        step=True,             # writes the step number to each line
        time=True,             # writes the time (in ps)
        potentialEnergy=True,  # writes potential energy of the system (KJ/mole)
        kineticEnergy=True,    # writes the kinetic energy of the system (KJ/mole)
        totalEnergy=True,      # writes the total energy of the system (KJ/mole)
        temperature=True,      # writes the temperature (in K)
        volume=True,           # writes the volume (in nm^3)
        density=True)          # writes the density (in g/mL)
    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(state_data_reporter)
    simulation.step(num_steps)
    print('Ran simulation. Temp =', temperature, 'pressure =', simulation.context.getParameter(MonteCarloBarostat.Pressure()) , 'Time = ',(num_steps*1)/1000000, 'ns')

simulation.saveState('eq_final.state')
simulation.saveCheckpoint('eq_final.chk')

#Save system for reinitialization if needed
system = simulation.context.getSystem()
with open('system_final.xml', 'w') as output:
    output.write(openmm.XmlSerializer.serialize(system))


