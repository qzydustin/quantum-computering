OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rz(-7*pi/8) q[1];
sx q[1];
cz q[0],q[1];