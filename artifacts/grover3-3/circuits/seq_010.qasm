OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rz(-3*pi/2) q[0];
sx q[1];
rz(-pi) q[1];