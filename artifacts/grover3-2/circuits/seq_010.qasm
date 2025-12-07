OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
sx q[0];
rz(pi/2) q[0];
sx q[1];