"""
Simon 3-qubit circuit generator.

Reference: QASMBench simon_n6 (simplified to 3 qubits)
Secret string: s = 10 (2-bit case)
"""

from __future__ import annotations

from typing import Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def simon_3q_circuit(secret_string: Optional[str] = None) -> QuantumCircuit:
    """Create a Simon's algorithm circuit for 3 qubits (2 input + 1 output)."""
    n_input = 2
    n_output = 1

    if secret_string is None:
        secret_string = "10"

    if len(secret_string) != n_input:
        raise ValueError(f"Secret string must be {n_input} bits long")

    input_reg = QuantumRegister(n_input, "input")
    output_reg = QuantumRegister(n_output, "output")
    creg = ClassicalRegister(n_input, "c")
    qc = QuantumCircuit(input_reg, output_reg, creg)

    # 1) Uniform superposition on input qubits
    for q in range(n_input):
        qc.h(input_reg[q])
    qc.barrier()

    # 2) Oracle
    qc.x(output_reg[0])
    if secret_string == "10":
        qc.cx(input_reg[1], output_reg[0])
        qc.ccx(input_reg[0], input_reg[1], output_reg[0])
    elif secret_string == "11":
        qc.cx(input_reg[0], output_reg[0])
        qc.cx(input_reg[1], output_reg[0])
    elif secret_string == "01":
        qc.cx(input_reg[0], output_reg[0])
    else:
        qc.cx(input_reg[1], output_reg[0])
        qc.ccx(input_reg[0], input_reg[1], output_reg[0])
    qc.barrier()

    # 3) Hadamard on input qubits
    for q in range(n_input):
        qc.h(input_reg[q])
    qc.barrier()

    # 4) Measure input qubits
    for i in range(n_input):
        qc.measure(input_reg[i], creg[i])

    return qc
