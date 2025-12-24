import importlib.util
ase_spec = importlib.util.find_spec("ase")
if(ase_spec is None):
  print("Cannot find ASE, this test will be skipped")
  exit()

from ase import Atoms
from pinq.calculator import PinqCalculator

alat =  3.567095

atoms = Atoms('C2', positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
atoms.set_cell([[0.0, alat/2.0, alat/2.0],
                [alat/2.0, 0.0, alat/2.0],
                [alat/2.0, alat/2.0, 0.0]], scale_atoms=True)
atoms.set_pbc(True)

atoms.calc = PinqCalculator(ecut = 70.0, extra_bands = 1)
atoms.calc.calculate(atoms)
energy = atoms.get_potential_energy()

print("Energy = ", energy);

assert abs(energy - -297.94410666135354) < 3.0e-5

import pinq

pinq.clear()

atoms.set_pbc([True, True, False])
pinq.ions.from_ase(atoms)
pinq.cell.status()
pinq.ions.status()


