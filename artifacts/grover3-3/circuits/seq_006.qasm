OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
sx q[0];
rz(-3*pi/2) q[0];