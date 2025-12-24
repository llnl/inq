#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

inq clear
inq cell orthorhombic 10.0 10.0 12.0 bohr
inq ions insert N 0 0 -1.1 bohr
inq ions insert N 0 0  1.1 bohr

inq electrons cutoff 40.0 Hartree
inq ground-state tolerance 1e-9
inq theory lda

inq run ground-state

inq util match `inq results ground-state energy total`             -20.642587680197 1e-5
inq util match `inq results ground-state energy kinetic`            13.163930063201 1e-5
inq util match `inq results ground-state energy eigenvalues`        -5.266573489001 1e-5
inq util match `inq results ground-state energy hartree`            14.494247686755 1e-5
inq util match `inq results ground-state energy external`          -39.444428447503 1e-5
inq util match `inq results ground-state energy non-local`          -1.620517298849 1e-5
inq util match `inq results ground-state energy xc`                 -5.718385218952 1e-5
inq util match `inq results ground-state energy nvxc`               -6.354053179360 1e-5
inq util match `inq results ground-state energy exact-exchange`      0.000000000000 1e-5
inq util match `inq results ground-state energy ion`                -1.517434464849 1e-5

inq util match `inq results ground-state forces 0`  -4.71156857800903538592e-09  3.27682700350424067102e-09  1.38951866717709787702e-01  3e-5
inq util match `inq results ground-state forces 1`  -8.72255760007185984415e-09 -9.93804962317393666478e-09 -1.38952071668748566857e-01  3e-5
#extra check to verify the indiviual component query
inq util match `inq results ground-state forces 1 z` -1.38952071668748566857e-01 3e-5

inq real-time time-step 0.025 atu
inq real-time num-steps 10
inq real-time ions ehrenfest

inq run real-time


