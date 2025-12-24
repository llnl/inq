#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

inq clear

inq cell orthorhombic 10.0 10.0 12.0 bohr

inq species N file `inq util test-data`/N_diagonal.upf.gz

inq ions insert N 0 0 -1.1 bohr
inq ions insert N 0 0  1.1 bohr

inq electrons cutoff 40.0 Hartree
inq ground-state tolerance 1e-9

inq run ground-state

printf "*****************************\n\n  Checking diagonal results\n\n*****************************\n"
inq util match `inq results ground-state energy total`           -20.762940910240 1e-5
inq util match `inq results ground-state energy kinetic`          13.232673106902 1e-5
inq util match `inq results ground-state energy eigenvalues`      -5.239327800413 1e-5
inq util match `inq results ground-state energy hartree`          14.564750154932 1e-5
inq util match `inq results ground-state energy external`        -39.564138921256 1e-5
inq util match `inq results ground-state energy non-local`        -1.623021810555 1e-5
inq util match `inq results ground-state energy xc`               -5.855768975414 1e-5
inq util match `inq results ground-state energy nvxc`             -6.414340485368 1e-5
inq util match `inq results ground-state energy exact-exchange`    0.000000000000 1e-5
inq util match `inq results ground-state energy ion`              -1.517434464849 1e-5

inq util match `inq results ground-state forces 0`  7.64111972132756120072e-08	-6.02887672156627078888e-08    1.43844275547036326568e-01  3e-5
inq util match `inq results ground-state forces 1`  1.15238467725916463107e-07	 1.06655760309057439416e-08 	-1.43848244252428980605e-01  3e-5

inq species N file `inq util test-data`/N_non_diagonal.upf.gz

inq run ground-state

printf "*********************************\n\n  Checking non-diagonal results\n\n*********************************\n"
inq util match `inq results ground-state energy total`           -20.762926512333 1e-5
inq util match `inq results ground-state energy kinetic`          13.233164631068 1e-5
inq util match `inq results ground-state energy eigenvalues`      -5.239282689379 1e-5
inq util match `inq results ground-state energy hartree`          14.564808824712 1e-5
inq util match `inq results ground-state energy external`        -39.564190987496 1e-5
inq util match `inq results ground-state energy non-local`        -1.623505362015 1e-5
inq util match `inq results ground-state energy xc`               -5.855769153752 1e-5
inq util match `inq results ground-state energy nvxc`             -6.414368620358 1e-5
inq util match `inq results ground-state energy exact-exchange`    0.000000000000 1e-5
inq util match `inq results ground-state energy ion`              -1.517434464849 1e-5

inq util match `inq results ground-state forces 0`   -1.42572049022417129734e-10  6.45807980741339751532e-11  1.44121353183289169220e-01  3e-4
inq util match `inq results ground-state forces 1`   -1.22189471061201702645e-10  6.08219338834519785562e-11 -1.44121353818859043727e-01  3e-4
