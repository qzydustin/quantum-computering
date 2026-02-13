OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rz(-5*pi/8) q[0];
sx q[0];