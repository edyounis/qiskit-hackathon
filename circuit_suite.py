'''


CODE BELOW PROVIDED BY TWO OTHER HACKATHON TEAMS.


suite of test circuits that can be provided as input to transformer_analysis


'''

import math
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

import numpy as np
from qiskit_chemistry.aqua_extensions.components.variational_forms import UCCSD

from qiskit import compile, BasicAer
from qiskit.circuit import Gate, InstructionSet
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import *

import itertools
from qiskit.qasm import pi

####################################################################################################################################
def ghz():
    qreg = QuantumRegister(3)
    circuit = QuantumCircuit(qreg, name='ghz')
    
    circuit.h(q[0])
    circuit.cx(q[0],q[1])
    circuit.cx(q[1],q[2])
    
    return circuit
####################################################################################################################################
def qft(number_qubits: int):
    """Create quantum fourier transform circuit on quantum register qreg."""
    qreg = QuantumRegister(number_qubits)
    circuit = QuantumCircuit(qreg, name="qft")

    for i in range(number_qubits):
        for j in range(i):
            circuit.cu1(math.pi / float(2 ** (i - j)), qreg[i], qreg[j])
        circuit.h(qreg[i])

    return circuit
####################################################################################################################################
def UCCSD_LiH_sto3g(active_occupied=[], active_unoccupied=[],
                    map_type='parity', two_qubit_reduction=False,
                    depth=1):

    # Set up problem-dependent variables related to molecular system and basis set
    n_orbitals = 12
    n_particles = 4
    n_qubits = 12

    if map_type == 'parity' and two_qubit_reduction:
        n_qubits -= 2

    # Define the variational #################################
    var_form = UCCSD(n_qubits, depth=depth,
                     num_orbitals=n_orbitals, num_particles=n_particles,
                     active_occupied=active_occupied, active_unoccupied=active_unoccupied,
                     qubit_mapping=map_type,two_qubit_reduction=two_qubit_reduction,
                     num_time_slices=1)

    # Arbitrary list of values for the variational parameters (does not impact circuit structure)
    params = np.ones(var_form._num_parameters)

    # Return the corresponding quantum circuit
    circuit = var_form.construct_circuit(params)

    print(circuit.width())
    print(circuit.depth())
    print(circuit.size())
    print(circuit.count_ops())

    return circuit
####################################################################################################################################
def ripple_adder(number_qubits: int):
    """
    A quantum adder circuit
    Finds an adder that uses at most number_qubits qubits.
    For odd number_qubits this means that one qubit is not used.
    """
    n = ((number_qubits - 2) // 2) * 2
    cin = QuantumRegister(1)
    a = QuantumRegister(n)
    cout = QuantumRegister(1)

    # Build a temporary subcircuit that adds a to b,
    # storing the result in b
    adder_circuit = QuantumCircuit(cin, a, cout, name="ripple_adder")
    adder_circuit.majority(cin[0], a[0], a[1])
    for j in range((n - 2)//2):
        adder_circuit.majority(a[2*j + 1], a[2*j + 2], a[2*j + 3])
    adder_circuit.cx(a[n - 1], cout[0])
    for j in reversed(range((n-2)//2)):
        adder_circuit.umaj_add(a[2*j + 1], a[2*j + 2], a[2*j + 3])
    adder_circuit.umaj_add(cin[0], a[0], a[1])

    return adder_circuit


class MajorityGate(Gate):
    def __init__(self, c, b, a, circ=None):
        """Create new Toffoli gate."""
        super().__init__("maj", [], [c, b, a], circ)

    def _define_decompositions(self):
        decomposition = DAGCircuit()
        q = QuantumRegister(3)
        decomposition.add_qreg(q)
        c, b, a = q
        rule = [
            CnotGate(a, b),
            CnotGate(a, c),
            ToffoliGate(c, b, a)
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        raise NotImplementedError("Inverse not implemented")

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.majority(self.qargs[0], self.qargs[1], self.qargs[2]))


def majority(self, c, b, a):
    """Apply a majority function on the given qubits"""
    if isinstance(c, QuantumRegister) and \
            isinstance(a, QuantumRegister) and \
            isinstance(b, QuantumRegister) and \
            len(c) == len(b) and len(b) == len(a):
        instructions = InstructionSet()
        for i in range(c.size):
            instructions.add(self.majority((c, i), (b, i), (a, i)))
        return instructions

    self._check_qubit(c)
    self._check_qubit(b)
    self._check_qubit(a)
    self._check_dups([c, b, a])
    return self._attach(MajorityGate(c, b, a, circ=self))


QuantumCircuit.majority = majority


class UnMajAdd(Gate):
    def __init__(self, a, b, c, circ=None):
        """Create new Unmajority and add (UMA) gate."""
        super().__init__("uma", [], [a, b, c], circ)

    def _define_decompositions(self):
        # Decomposition with minimal nr of CNOTs
        decomposition_cnot = DAGCircuit()
        q = QuantumRegister(3)
        a, b, c = q
        decomposition_cnot.add_qreg(q)
        rule = [
            ToffoliGate(a, b, c),
            CnotGate(c, a),
            CnotGate(a, b)
        ]
        for inst in rule:
            decomposition_cnot.apply_operation_back(inst)

        # Decomposition with minimal depth
        decomposition_par = DAGCircuit()
        q = QuantumRegister(3)
        a, b, c = q
        decomposition_par.add_qreg(q)
        rule = [
            XGate(b),
            CnotGate(a, b),
            ToffoliGate(a, b, c),
            XGate(b),
            CnotGate(c, a),
            CnotGate(c, b)
        ]
        for inst in rule:
            decomposition_par.apply_operation_back(inst)
        self._decompositions = [decomposition_cnot, decomposition_par]

    def inverse(self):
        """Invert this gate."""
        raise NotImplementedError("Inverse not implemented")

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.umaj_add(self.qargs[0], self.qargs[1], self.qargs[2]))


def umaj_add(self, a, b, c):
    """Apply a majority function on the given qubits"""
    if isinstance(c, QuantumRegister) and \
            isinstance(a, QuantumRegister) and \
            isinstance(b, QuantumRegister) and \
            len(c) == len(b) and len(b) == len(a):
        instructions = InstructionSet()
        for i in range(a.size):
            instructions.add(self.umaj_add((a, i), (b, i), (c, i)))
        return instructions

    self._check_qubit(c)
    self._check_qubit(b)
    self._check_qubit(a)
    self._check_dups([c, b, a])
    return self._attach(UnMajAdd(a, b, c, circ=self))


QuantumCircuit.umaj_add = umaj_add
####################################################################################################################################
def toffoli(number_qubits: int):
    assert number_qubits >= 2
    q = QuantumRegister(number_qubits)
    qc = QuantumCircuit(q, name="toffoli")
    # for i in range(number_qubits-1):
    #     qc.h(controls[i])
    qc.ntoffoli(q[number_qubits-1], *q[0:number_qubits-1])
    # qc.measure(controls, c_controls)
    # qc.measure(target, c_target)
    return qc

class NcrxGate(Gate):
    """n-controlled x rotation gate."""

    def __init__(self, theta, tgt, *ctls, circ=None):
        """Create new Toffoli gate."""
        assert len(ctls) >= 1
        super().__init__(f"c^{len(ctls)}rx", [theta], [tgt] + list(ctls), circ)

    def _define_decompositions(self):
        decomposition = DAGCircuit()
        nr_qubits = len(self.qargs)
        q = QuantumRegister(nr_qubits)
        last_control = q[1]
        target = q[0]
        decomposition.add_qreg(q)
        if nr_qubits == 2:
            # Equal to crx of theta
            crx_theta = Cu3Gate(self.params[0], -pi/2, pi/2, last_control, target)
            decomposition.apply_operation_back(crx_theta)
        else:
            # Recurse
            rule = [
                # C-sqrt(rx(theta)) gate
                Cu3Gate(self.params[0]/2, -pi/2, pi/2, last_control, target),
                NcrxGate(pi, last_control, *q[2:]), # toffoli
                Cu3Gate(self.params[0]/2, -pi/2, pi/2, last_control, target).inverse(),
                NcrxGate(pi, last_control, *q[2:]), # toffoli
                NcrxGate(self.params[0]/2, target, *q[2:]) # c^nrx(theta/2) gate on n-1 qubits
            ]
            for inst in rule:
                decomposition.apply_operation_back(inst)
            # decomposition.apply_operation_back(ToffoliGate(q[1], q[2], q[0]))
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.ncrx(self.params[0], self.qargs[0], *self.qargs[1:]))

def ncrx(self, theta, tgt, *ctls):
    """Apply n-controlled x-rotation(theta) to target from controls"""
    if all(isinstance(ctl, QuantumRegister) for ctl in ctls) and \
            isinstance(tgt, QuantumRegister) and \
            all(len(ctl) == len(tgt) for ctl in ctls):
        instructions = InstructionSet()
        for i in range(ctls[0].size):
            instructions.add(self.ntoffoli(theta, (tgt, i), *zip(ctls, itertools.repeat(i))))
        return instructions

    for ctl in ctls:
        self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups(list(ctls) + [tgt])
    return self._attach(NcrxGate(theta, tgt, *ctls, circ=self))

def ntoffoli(self, tgt, *ctls):
    """Apply n-controlled Toffoli to tgt with controls."""
    if all(isinstance(ctl, QuantumRegister) for ctl in ctls) and \
            isinstance(tgt, QuantumRegister) and \
            all(len(ctl) == len(tgt) for ctl in ctls):
        instructions = InstructionSet()
        for i in range(ctls[0].size):
            instructions.add(self.ntoffoli((tgt, i), *zip(ctls, itertools.repeat(i))))
        return instructions

    for ctl in ctls:
        self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups(list(ctls) + [tgt])
    return self._attach(NcrxGate(pi, tgt, *ctls, circ=self))

QuantumCircuit.ncrx = ncrx
QuantumCircuit.ntoffoli = ntoffoli
####################################################################################################################################