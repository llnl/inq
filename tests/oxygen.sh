#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

distance=121.0 #pm

inq clear
inq cell cubic 10.0 bohr finite
inq ions insert O -$distance/2 0 0 pm
inq ions insert O  $distance/2 0 0 pm

inq electrons extra states 2
inq electrons spacing 0.43 bohr
inq electrons spin polarized
inq electrons temperature 1000 K

inq ground-state mixing 0.2
inq ground-state tolerance 1e-8

inq run ground-state

inq util match `inq results ground-state energy total`          -32.950379793895 1e-5
inq util match `inq results ground-state energy kinetic`         20.949203117436 1e-5
inq util match `inq results ground-state energy eigenvalues`     -7.232665451933 1e-5
inq util match `inq results ground-state energy hartree`         42.217542391170 1e-5
inq util match `inq results ground-state energy external`       -99.196829409969 1e-5
inq util match `inq results ground-state energy non-local`       -4.469732043222 1e-5
inq util match `inq results ground-state energy xc`              -8.194679214989 1e-5
inq util match `inq results ground-state energy nvxc`            -8.950391898517 1e-5
inq util match `inq results ground-state energy exact-exchange`   0.000000000000 1e-5
inq util match `inq results ground-state energy ion`             15.744115365679 1e-5

#do both checks to make sure both query modes are working
inq util match  `inq results ground-state magnetization`  0.000000000000 0.000000000000 1.999999000000  1e-5
inq util match `inq results ground-state magnetization z` 1.999999000000 1e-5
