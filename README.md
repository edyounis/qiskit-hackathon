# qiskit-hackathon

##Abstract
Minimizing circuit depth is one of the main metrics of compiler design, however, achieving shorter circuit depth does not necessarily mean the result is more robust to noise. Investigating the effect of noise on the result should be one of the main objectives of compilers. Therefore, it is very important to evaluate compiler quality by investigating the correctness of the final result on potential quantum hardware. We do this by using some quantum circuits as benchmark, injecting noise to the transformed circuit, and measuring the fidelity of the final process. Furthermore, this process can be generalized to any circuit transformation such as error correcting codes.
