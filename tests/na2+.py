
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

assert pinq.util.match(pinq.results.ground_state.energy.total(),            -0.643875190035, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.kinetic(),           0.018878577118, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.eigenvalues(),      -0.089771561959, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.hartree(),           0.000453780325, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.external(),          0.076259198549, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.non_local(),         0.002713603553, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.xc(),               -0.376876179126, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.nvxc(),             -0.188530501831, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.exact_exchange(),    0.000000000000, 3e-5)
assert pinq.util.match(pinq.results.ground_state.energy.ion(),              -0.365304170454, 3e-5)
  
assert pinq.util.match(pinq.results.ground_state.forces()[0],  [3.74786824263793630366e-11, -1.05853471242772412887e-10,  1.31008348141167883100e-03], 3e-5)
assert pinq.util.match(pinq.results.ground_state.forces()[1],  [6.54160820496192356550e-11, -6.38554194489870691095e-11, -1.31008351907754708524e-03], 3e-5)

pinq.real_time.num_steps(3000)
pinq.real_time.time_step(0.075, "atu")
pinq.real_time.observables.dipole()
pinq.perturbations.kick([0, 0, 0.01])

pinq.run.real_time()

energy = pinq.results.real_time.total_energy()

assert pinq.util.match(energy[   0], -0.6438225831700184, 3e-5)
assert pinq.util.match(energy[  10], -0.6438225831586412, 3e-5)
assert pinq.util.match(energy[ 300], -0.6438225835346540, 3e-5)
assert pinq.util.match(energy[ 444], -0.6438225837767503, 3e-5)
assert pinq.util.match(energy[1000], -0.6438225844307714, 3e-5)
assert pinq.util.match(energy[1663], -0.6438225852107715, 3e-5)
assert pinq.util.match(energy[2000], -0.6438225854501384, 3e-5)
assert pinq.util.match(energy[2101], -0.6438225856304741, 3e-5)
assert pinq.util.match(energy[2748], -0.6438225861809898, 3e-5)
assert pinq.util.match(energy[2937], -0.6438225863718710, 3e-5)
assert pinq.util.match(energy[3000], -0.6438225864372554, 3e-5)

dipole = pinq.results.real_time.dipole()

assert pinq.util.match(dipole[   0], [-3.97681048e-09, -7.09794354e-09, -1.32841626e-08], 3e-5)
assert pinq.util.match(dipole[  10], [-4.97442347e-09, -7.24474784e-09, -6.82902345e-04], 3e-5)
assert pinq.util.match(dipole[ 300], [-4.12649853e-09, -4.94520268e-09,  9.28946989e-04], 3e-5)
assert pinq.util.match(dipole[ 444], [ 6.56710319e-09,  4.10472422e-09,  2.49459196e-05], 3e-5)
assert pinq.util.match(dipole[1000], [-1.68405742e-09, -1.74218890e-09,  6.43184391e-04], 3e-5)
assert pinq.util.match(dipole[1663], [ 3.70110317e-09,  2.40236705e-09,  1.14428514e-04], 3e-5)
assert pinq.util.match(dipole[2000], [ 3.89065598e-09,  6.58938917e-09, -3.79885931e-05], 3e-5)
assert pinq.util.match(dipole[2101], [-2.46808191e-09, -6.44963568e-09,  2.51021618e-04], 3e-5)
assert pinq.util.match(dipole[2748], [ 7.49161622e-09,  6.63569241e-09,  7.68891245e-04], 3e-5)
assert pinq.util.match(dipole[2937], [-2.46748677e-09, -5.12091893e-09,  9.61352605e-04], 3e-5)
assert pinq.util.match(dipole[3000], [ 6.42990291e-09,  5.12773025e-09,  8.26916447e-04], 3e-5)
