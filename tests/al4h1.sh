#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

inq clear
inq cell cubic 7.6524459 bohr

inq ions insert fractional Al  0.0  0.0  0.0
inq ions insert fractional Al  0.0  0.5  0.5
inq ions insert fractional Al  0.5  0.0  0.5 
inq ions insert fractional Al  0.5  0.5  0.0
inq ions insert fractional H   0.1  0.2  0.3 

inq electrons cutoff 30 hartree
inq electrons extra states 1
inq electrons temperature 300 Kelvin
inq theory pbe
inq kpoints shifted grid 2 2 2
inq ground-state tolerance 1e-8

inq run ground-state

inq util match `inq results ground-state energy total`             -9.802338589909  3e-5
inq util match `inq results ground-state energy kinetic`            4.200431506537  3e-5
inq util match `inq results ground-state energy eigenvalues`        0.602436655807  3e-5
inq util match `inq results ground-state energy hartree`            0.219185988851  3e-5
inq util match `inq results ground-state energy external`          -0.562805233731  3e-5
inq util match `inq results ground-state energy non-local`          1.427216752802  3e-5
inq util match `inq results ground-state energy xc`                -4.767995491137  3e-5
inq util match `inq results ground-state energy nvxc`              -4.900778347503  3e-5
inq util match `inq results ground-state energy exact-exchange`     0.000000000000  3e-5
inq util match `inq results ground-state energy ion`              -10.318372113231  3e-5

inq util match `inq results ground-state forces 0`                 -2.09878683861885427520e-02 -3.86439030449507309184e-02 -4.92550174029726950398e-02  3e-5
inq util match `inq results ground-state forces 1`                 -2.09841584974472528113e-02  4.92322743523460754078e-02  3.86387752796131661581e-02  3e-5
inq util match `inq results ground-state forces 2`                  6.67547441607678308795e-03 -1.28302905479014666551e-02  1.28262908762605074703e-02  3e-5
inq util match `inq results ground-state forces 3`                  7.67798179337165348501e-03  1.25067841618366794310e-02 -1.25106042421354955035e-02  3e-5
inq util match `inq results ground-state forces 4`                  2.76562191520605543671e-02 -1.01874661145976881660e-02  1.03629838578964024454e-02  3e-5

inq electrons extra states 0
inq real-time num-steps 30
inq real-time time-step 0.055 atu

inq run real-time

inq util match `inq results real-time total-steps`                 30    1e-16
inq util match `inq results real-time total-time`                   1.65 1e-16
inq util match `inq results real-time total-energy  0`             -9.80233858990913375919e+00 3e-5
inq util match `inq results real-time total-energy 10`             -9.80233858990918349718e+00 3e-5
inq util match `inq results real-time total-energy 20`             -9.80233858990920126075e+00 3e-5
inq util match `inq results real-time total-energy 30`             -9.80233858990928652588e+00 3e-5

