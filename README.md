# Benchmarking Circuit Transformations using Noise Injection

## Abstract

Minimizing circuit depth is one of the main metrics of compiler design, however, achieving shorter circuit depth does not necessarily mean the result is more robust to noise. Investigating the effect of noise on the result should be one of the main objectives of compilers. Therefore, it is very important to evaluate compiler quality by investigating the correctness of the final result on potential quantum hardware. We do this by using some quantum circuits as benchmark, injecting noise to the transformed circuit, and measuring the fidelity of the final process. Furthermore, this process can be generalized to any circuit transformation such as error correcting codes.

We include some circuits developed by the Writing Standardized Benchmarks team and the Library of Quantum Circuits for Arithmetic team.

## Future Work

1. Adding a tranpiler pass to return the final layout of the circuit to the original layout.
2. Modify the unitary simulator to sample noise models and incorporate Ignis functions.
