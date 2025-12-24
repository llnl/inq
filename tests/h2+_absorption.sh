#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

inq clear
inq cell orthorhombic 8 8 10 bohr finite

inq ions insert H 0.0 0.0 -1.0 bohr
inq ions insert H 0.0 0.0  1.0 bohr

inq electrons extra-electrons -1
inq electrons spacing 0.45 bohr
inq ground-state tolerance 1e-8

inq theory non-interacting

inq run ground-state

inq util match `inq results ground-state energy total`            -0.602091471326 1e-5
inq util match `inq results ground-state energy kinetic`           0.578156580803 1e-5
inq util match `inq results ground-state energy eigenvalues`      -1.102091471326 1e-5
inq util match `inq results ground-state energy hartree`           0.000000000000 1e-5
inq util match `inq results ground-state energy external`         -1.592916754227 1e-5
inq util match `inq results ground-state energy non-local`        -0.087331297902 1e-5
inq util match `inq results ground-state energy xc`                0.000000000000 1e-5
inq util match `inq results ground-state energy nvxc`              0.000000000000 1e-5
inq util match `inq results ground-state energy exact-exchange`    0.000000000000 1e-5
inq util match `inq results ground-state energy ion`               0.500000000000 1e-5

inq rt num-steps 2000
inq rt time-step 0.07 atu
inq rt observables dipole
inq perturbation kick 0 0 0.01

inq run rt

inq util match `inq results rt total-steps`               2000    1e-16
inq util match `inq results rt total-time`                140.00  1e-12

inq util match `inq results rt total-energy    0`          -6.02038713767435340607e-01 3e-5
inq util match `inq results rt total-energy  250`          -6.02039400615363429203e-01 3e-5
inq util match `inq results rt total-energy  500`          -6.02039563777437947145e-01 3e-5
inq util match `inq results rt total-energy  750`          -6.02039650569539008806e-01 3e-5
inq util match `inq results rt total-energy 1000`          -6.02039708902089576448e-01 3e-5
inq util match `inq results rt total-energy 1250`          -6.02039752399125771554e-01 3e-5
inq util match `inq results rt total-energy 1500`          -6.02039786801750076428e-01 3e-5
inq util match `inq results rt total-energy 1750`          -6.02039815142913492529e-01 3e-5
inq util match `inq results rt total-energy 2000`          -6.02039839201153115233e-01 3e-5

#check the individual component query works too
inq util match `inq results rt dipole    0 x`              -1.54124808281475814305e-03 3e-5
inq util match `inq results rt dipole    0 y`              -1.54124808014932277930e-03 3e-5

inq util match `inq results rt dipole    0`                -1.54124808281475814305e-03	-1.54124808014932277930e-03	-3.18539677983979620680e-04 3e-5
inq util match `inq results rt dipole  250`                -1.53715512837854792819e-03	-1.53715513055107852979e-03	-2.20572792225880963402e-02 3e-5
inq util match `inq results rt dipole  500`                -1.53906074142993488929e-03	-1.53906073191713266800e-03	-9.61422811969275641075e-03 3e-5
inq util match `inq results rt dipole  750`                -1.54546517863648428387e-03	-1.54546518301039068587e-03	 1.73412993574253576634e-02 3e-5
inq util match `inq results rt dipole 1000`                -1.54496267187288608726e-03	-1.54496265723247784389e-03  1.65933169828854504280e-02 3e-5
inq util match `inq results rt dipole 1250`                -1.54117965601463578589e-03	-1.54117965838824013715e-03	-1.06809681678022728307e-02 3e-5
inq util match `inq results rt dipole 1500`                -1.54114528104081047362e-03	-1.54114527851972925657e-03	-2.18011257603510054393e-02 3e-5
inq util match `inq results rt dipole 1750`                -1.54132025443926706787e-03	-1.54132024461676292969e-03	 9.09655537626315527366e-04 3e-5
inq util match `inq results rt dipole 2000`                -1.53949216003116339489e-03	-1.53949217012619974689e-03	 2.16152517670456442711e-02 3e-5
