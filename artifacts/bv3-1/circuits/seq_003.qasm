OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
sx q[0];
rz(pi) q[0];
cz q[1],q[0];