# Gate 2-C interval coverage

Intervals use validation residuals only. OOD coverage is empirical; it is not presented as a distribution-free guarantee.

| Protocol | Method | Nominal | Record coverage | Identity/crossed coverage | Width (eV) | Status |
|---|---|---:|---:|---:|---:|---|
| iid | identity | 80% | 0.8133 | 0.8180 | 0.2761 | ATTAINABLE |
| iid | identity | 90% | 0.9021 | 0.9052 | 0.3752 | ATTAINABLE |
| iid | identity | 95% | 0.9577 | 0.9585 | 0.5040 | ATTAINABLE |
| iid | record | 80% | 0.8137 | 0.8185 | 0.2768 | ATTAINABLE |
| iid | record | 90% | 0.9025 | 0.9057 | 0.3776 | ATTAINABLE |
| iid | record | 95% | 0.9577 | 0.9585 | 0.5040 | ATTAINABLE |
| iid | structure | 80% | 0.8133 | 0.8180 | 0.2761 | ATTAINABLE |
| iid | structure | 90% | 0.9021 | 0.9052 | 0.3752 | ATTAINABLE |
| iid | structure | 95% | 0.9577 | 0.9585 | 0.5040 | ATTAINABLE |
| donor_cold | identity | 80% | 0.9996 | 0.9996 | 0.9134 | ATTAINABLE |
| donor_cold | identity | 90% | 1.0000 | 1.0000 | 1.0167 | ATTAINABLE |
| donor_cold | identity | 95% | NA | NA | NA | UNATTAINABLE_FINITE_SAMPLE |
| donor_cold | record | 80% | 0.8476 | 0.8333 | 0.3045 | ATTAINABLE |
| donor_cold | record | 90% | 0.9240 | 0.9160 | 0.4113 | ATTAINABLE |
| donor_cold | record | 95% | 0.9680 | 0.9639 | 0.5103 | ATTAINABLE |
| donor_cold | structure | 80% | 0.8476 | 0.8333 | 0.3045 | ATTAINABLE |
| donor_cold | structure | 90% | 0.9231 | 0.9153 | 0.4106 | ATTAINABLE |
| donor_cold | structure | 95% | 0.9680 | 0.9639 | 0.5103 | ATTAINABLE |
| acceptor_cold | identity | 80% | 0.9937 | 0.9942 | 0.8564 | ATTAINABLE |
| acceptor_cold | identity | 90% | 0.9960 | 0.9964 | 0.9396 | ATTAINABLE |
| acceptor_cold | identity | 95% | 0.9960 | 0.9964 | 0.9930 | ATTAINABLE |
| acceptor_cold | record | 80% | 0.7988 | 0.7959 | 0.3042 | ATTAINABLE |
| acceptor_cold | record | 90% | 0.9093 | 0.9108 | 0.4304 | ATTAINABLE |
| acceptor_cold | record | 95% | 0.9562 | 0.9587 | 0.5543 | ATTAINABLE |
| acceptor_cold | structure | 80% | 0.7997 | 0.7969 | 0.3045 | ATTAINABLE |
| acceptor_cold | structure | 90% | 0.9097 | 0.9112 | 0.4329 | ATTAINABLE |
| acceptor_cold | structure | 95% | 0.9566 | 0.9592 | 0.5555 | ATTAINABLE |
| pair_cold | identity | 80% | 0.7995 | 0.8024 | 0.2693 | ATTAINABLE |
| pair_cold | identity | 90% | 0.8844 | 0.8867 | 0.3628 | ATTAINABLE |
| pair_cold | identity | 95% | 0.9476 | 0.9482 | 0.4781 | ATTAINABLE |
| pair_cold | record | 80% | 0.8016 | 0.8046 | 0.2711 | ATTAINABLE |
| pair_cold | record | 90% | 0.8865 | 0.8885 | 0.3648 | ATTAINABLE |
| pair_cold | record | 95% | 0.9476 | 0.9482 | 0.4781 | ATTAINABLE |
| pair_cold | structure | 80% | 0.7995 | 0.8024 | 0.2693 | ATTAINABLE |
| pair_cold | structure | 90% | 0.8844 | 0.8867 | 0.3628 | ATTAINABLE |
| pair_cold | structure | 95% | 0.9480 | 0.9487 | 0.4795 | ATTAINABLE |
| both_cold | acceptor_identity_sensitivity | 80% | 0.9847 | 0.9846 | 0.5144 | ATTAINABLE |
| both_cold | acceptor_identity_sensitivity | 90% | 0.9932 | 0.9932 | 0.6306 | ATTAINABLE |
| both_cold | acceptor_identity_sensitivity | 95% | 0.9966 | 0.9966 | 0.7397 | ATTAINABLE |
| both_cold | donor_identity_sensitivity | 80% | 0.9915 | 0.9914 | 0.6086 | ATTAINABLE |
| both_cold | donor_identity_sensitivity | 90% | 0.9966 | 0.9966 | 0.7331 | ATTAINABLE |
| both_cold | donor_identity_sensitivity | 95% | 0.9983 | 0.9983 | 0.8866 | ATTAINABLE |
| both_cold | record | 80% | 0.8416 | 0.8416 | 0.2962 | ATTAINABLE |
| both_cold | record | 90% | 0.9370 | 0.9370 | 0.4065 | ATTAINABLE |
| both_cold | record | 95% | 0.9744 | 0.9744 | 0.4907 | ATTAINABLE |
| both_cold | structure | 80% | 0.8330 | 0.8332 | 0.2910 | ATTAINABLE |
| both_cold | structure | 90% | 0.9353 | 0.9353 | 0.3987 | ATTAINABLE |
| both_cold | structure | 95% | 0.9779 | 0.9778 | 0.4963 | ATTAINABLE |
| full_scaffold_cold | identity | 80% | 0.9258 | 0.9255 | 0.4069 | ATTAINABLE |
| full_scaffold_cold | identity | 90% | 0.9653 | 0.9649 | 0.5319 | ATTAINABLE |
| full_scaffold_cold | identity | 95% | 0.9853 | 0.9843 | 0.6417 | ATTAINABLE |
| full_scaffold_cold | record | 80% | 0.8307 | 0.8283 | 0.2784 | ATTAINABLE |
| full_scaffold_cold | record | 90% | 0.9191 | 0.9182 | 0.3961 | ATTAINABLE |
| full_scaffold_cold | record | 95% | 0.9587 | 0.9587 | 0.5092 | ATTAINABLE |
| full_scaffold_cold | structure | 80% | 0.8280 | 0.8254 | 0.2764 | ATTAINABLE |
| full_scaffold_cold | structure | 90% | 0.9182 | 0.9170 | 0.3954 | ATTAINABLE |
| full_scaffold_cold | structure | 95% | 0.9587 | 0.9587 | 0.5090 | ATTAINABLE |
