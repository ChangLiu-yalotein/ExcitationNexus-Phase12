# Gate 1-C1 Geometry Value

All 372 duplicate groups: geometry RMSD versus primary target range Spearman = `0.8898125862176925`.

Same role-aware identity subset: `153` groups, Spearman = `0.8628486755391407`, p = `1.3546799052117898e-46`.

Validation-only counterfactuals used six frozen checkpoints and `2309` records; no test counterfactual was run. Ordered noise response: `True`. Frozen val/test prediction dispersion covers `238` duplicate groups; train duplicates were not newly inferred.

This correlation is not evidence that larger 3D models will outperform the 2D baseline: absolute duplicate-label ranges are small and 219/372 duplicate groups change role-aware identity. The 153 same-role-aware groups were therefore reported separately.

## Validation-only counterfactuals

|Condition|Merged mean abs prediction change|DAU mean abs prediction change|
|---|---:|---:|
|coordinates set to zero|0.154220|0.114412|
|Gaussian noise 0.01 A|0.000114|0.000095|
|Gaussian noise 0.05 A|0.000569|0.000490|
|Gaussian noise 0.10 A|0.001148|0.000988|
|D/A separation 0.10 A|0.000432|~0|
|D/A separation 0.50 A|0.002000|~0|
|D/A separation 1.00 A|0.003533|~0|

Global rotation/translation changes predictions by only about `1e-8 eV`. Both models use local distance information, but DAU is structurally blind to donor/acceptor relative displacement because its role subgraphs are processed independently before pooling. This is an architecture limitation, not evidence that geometry is irrelevant.
