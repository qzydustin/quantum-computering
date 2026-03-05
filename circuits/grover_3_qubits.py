"""
Grover 3-qubit circuit generator.

Reference: IBM Quantum Learning – Grover's algorithm
"""

from __future__ import annotations

from typing import Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def grover_3q_circuit(marked_states: Optional[list] = None) -> QuantumCircuit:
    """Create a 3-qubit Grover circuit with ancilla-based phase kickback oracle."""
    n = 3
    if marked_states is None:
        marked_states = [7]

    inp = QuantumRegister(n, "inp")
    anc = QuantumRegister(1, "anc")
    creg = ClassicalRegister(n, "c")
    qc = QuantumCircuit(inp, anc, creg)

    # Initialization: uniform superposition + ancilla to |->
    for q in range(n):
        qc.h(inp[q])
    qc.x(anc[0])
    qc.h(anc[0])
    qc.barrier()

    def apply_oracle(bitstring: str):
        bits = bitstring[::-1]
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(inp[i])
        qc.mcx([inp[i] for i in range(n)], anc[0])
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(inp[i])
        qc.barrier()

    def apply_diffusion():
        for q in range(n):
            qc.h(inp[q])
        for q in range(n):
            qc.x(inp[q])
        qc.h(inp[-1])
        qc.mcx([inp[i] for i in range(n - 1)], inp[-1])
        qc.h(inp[-1])
        for q in range(n):
            qc.x(inp[q])
        for q in range(n):
            qc.h(inp[q])
        qc.barrier()

    # Grover iteration
    for state in marked_states:
        apply_oracle(format(state, f"0{n}b"))
    apply_diffusion()

    # Clear ancilla
    qc.h(anc[0])
    qc.x(anc[0])
    qc.barrier()

    for i in range(n):
        qc.measure(inp[i], creg[i])
    return qc
