import importlib.util
ase_spec = importlib.util.find_spec("ase")
if(ase_spec is None):
  print("Cannot find ASE, this test will be skipped")
  exit()

from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.nwchem import NWChem
from ase.io import write
import _pinq

atoms = Atoms('H2', positions=[[0, 0, 0], [0.0, 0, 0.7]], cell = [3.0, 3.0, 4.0])
atoms.set_pbc(True)

atoms.calc = _pinq.calculator(ecut = 90.0, xc = 'LDA')
atoms.calc.calculate(atoms)
energy = atoms.get_potential_energy()


print("Energy = ", energy);

assert abs(energy - -32.62193464578265) < 3.0e-5

forces = atoms.get_forces()
print("Forces\n:", forces);

assert abs(forces[0, 0]) < 3.0e-5
assert abs(forces[0, 1]) < 3.0e-5
assert abs(forces[0, 2] - -4.73394384e+00) < 3.0e-5
assert abs(forces[1, 0]) < 3.0e-5
assert abs(forces[1, 1]) < 3.0e-5
assert abs(forces[1, 2] -  4.73394369e+00) < 3.0e-5

#Run a second time to check it works

atoms.calc.calculate(atoms)
energy = atoms.get_potential_energy()

print("Energy = ", energy);

assert abs(energy - -32.62193464578265) < 3.0e-5
