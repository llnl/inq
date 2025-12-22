
import pinq

pinq.clear()

dist = 2.01

pinq.cell.cubic(4.0, "angstrom")
pinq.cell.status()

pinq.ions.insert("Na", [0.0, 0.0, -dist/2.0], "angstrom")
pinq.ions.insert("Na", [0.0, 0.0,  dist/2.0], "angstrom")

pinq.species.file("Na", pinq.util.test_data() + "/Na.pz-n-vbc.UPF");

pinq.electrons.cutoff(24.0, "Hartree")
pinq.electrons.extra_electrons(-1)
pinq.ground_state.tolerance(1e-8)

pinq.run.ground_state()

assert pinq.util.match(pinq.results.ground_state.energy.total(),            -0.643814287043, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.kinetic(),           0.018925591948, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.eigenvalues(),      -0.089705264331, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.hartree(),           0.000453291561, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.external(),          0.076260504480, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.non_local(),         0.002721827596, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.xc(),               -0.376871332174, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.nvxc(),             -0.188519771478, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.exact_exchange(),    0.000000000000, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.ion(),              -0.365304170454, 3e-5)
  
assert pinq.util.match(pinq.results.ground_state.forces()[0],  [3.74786824263793630366e-11, -1.05853471242772412887e-10,  1.31008348141167883100e-03], 3e-5)
assert pinq.util.match(pinq.results.ground_state.forces()[1],  [6.54160820496192356550e-11, -6.38554194489870691095e-11, -1.31008351907754708524e-03], 3e-5)

pinq.real_time.num_steps(100)
pinq.real_time.time_step(0.0666, "atu")
pinq.real_time.observables.dipole()
pinq.perturbations.kick([0, 0, 0.01])

pinq.run.real_time()

energy = pinq.results.real_time.total_energy()

print(energy[  0])
print(energy[ 10])
print(energy[ 20])
print(energy[ 30])
print(energy[ 40])
print(energy[ 50])
print(energy[ 60])
print(energy[ 70])
print(energy[ 80])
print(energy[ 90])
print(energy[100])

assert pinq.util.match(energy[  0], -0.6437616782708677, 3e-5)
assert pinq.util.match(energy[ 10], -0.6437616782611675, 3e-5)
assert pinq.util.match(energy[ 20], -0.6437616782345381, 3e-5)
assert pinq.util.match(energy[ 30], -0.6437616781927767, 3e-5)
assert pinq.util.match(energy[ 40], -0.6437616782666558, 3e-5)
assert pinq.util.match(energy[ 50], -0.6437616782109121, 3e-5)
assert pinq.util.match(energy[ 60], -0.6437616782819332, 3e-5)
assert pinq.util.match(energy[ 70], -0.6437616782430907, 3e-5)
assert pinq.util.match(energy[ 80], -0.6437616782757755, 3e-5)
assert pinq.util.match(energy[ 90], -0.643761678221251 , 3e-5)
assert pinq.util.match(energy[100], -0.6437616783189151, 3e-5)

dipole = pinq.results.real_time.dipole()

print(dipole[  0])
print(dipole[ 10])
print(dipole[ 20])
print(dipole[ 30])
print(dipole[ 40])
print(dipole[ 50])
print(dipole[ 60])
print(dipole[ 70])
print(dipole[ 80])
print(dipole[ 90])
print(dipole[100])

assert pinq.util.match(dipole[  0], [-0.17853907, -0.17853954, -0.22782616], 3e-5)
assert pinq.util.match(dipole[ 10], [-0.17853874, -0.17853918, -0.22725359], 3e-5)
assert pinq.util.match(dipole[ 20], [-0.17853824, -0.17853867, -0.22702007], 3e-5)
assert pinq.util.match(dipole[ 30], [-0.17854107, -0.17854149, -0.22730725], 3e-5)
assert pinq.util.match(dipole[ 40], [-0.17854423, -0.17854459, -0.22806705], 3e-5)
assert pinq.util.match(dipole[ 50], [-0.17854279, -0.17854307, -0.22862196], 3e-5)
assert pinq.util.match(dipole[ 60], [-0.17854023, -0.17854046, -0.22864814], 3e-5)
assert pinq.util.match(dipole[ 70], [-0.17853626, -0.17853641, -0.22803839], 3e-5)
assert pinq.util.match(dipole[ 80], [-0.17853042, -0.17853045, -0.22719715], 3e-5)
assert pinq.util.match(dipole[ 90], [-0.17853408, -0.17853402, -0.22682688], 3e-5)
assert pinq.util.match(dipole[100], [-0.17854365, -0.17854348, -0.22715156], 3e-5)











