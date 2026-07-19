# Gate 1-B2 3D baseline admission smoke

Status: **GATE1B2_READY_FOR_FULL_TRAINING**.

The complete target-free DFT registry contains 15,016 records, 3,524,839 atoms, and 3,738,352 source bonds. The local 43 MiB graph cache is bound by SHA-256 `d7ca8d4ad6dda694e202e0ff94f5822d6b72404f8e5997acf986ee45f0202aa5` and is excluded from GitHub. Test targets and final673 were not read.

M3-Merged and M3-DAU-Shared use the same distance-only invariant backbone family, hidden width, layers, cutoff, optimizer, loss, seed, and target-blind 512/128 train/val subsets. Parameter counts are 36,689 and 36,461, a 0.621% difference. DAU invokes one shared-weight backbone separately on D/A/U subgraphs; it is not a legacy B2-1 reproduction.

| Admission evidence | M3-Merged | M3-DAU-Shared |
|---|---:|---:|
| Physical GPU | 0 | 1 |
| Two-batch losses | 0.8691, 0.5544 | 0.6930, 0.5187 |
| Parameter updated | yes | yes |
| Peak allocated GPU memory | 81.13 MiB | 76.19 MiB |
| Three-epoch wall time | 370.06 s | 373.24 s |
| Approx. train graphs/s | 4.151 | 4.115 |
| Max translation/rotation delta | 3.28e-7 | 1.79e-7 |
| Epoch 1/2/3 train loss | 0.4124 / 0.4048 / 0.3897 | 0.4249 / 0.4105 / 0.4187 |
| Epoch 1/2/3 val group-macro MAE | 0.1413 / 0.1319 / 0.1325 | 0.1427 / 0.1374 / 0.1396 |

All losses, outputs, and gradients were finite. These three epochs are plumbing admission only: the validation values are not used to rank architectures, select checkpoints, or freeze Gate 1-B3 hyperparameters. Runtime shows the present CPU radius-graph construction bottleneck; DAU's three backbone calls produced about 0.9% more wall time in this small smoke, while induced subgraphs reduced its allocated GPU memory.

Original explicit roles remain primary. Of 387 empty-donor+unknown records, 198 have one unique/symmetry-equivalent graph-mapped role set and 189 remain `UNRESOLVED_AMBIGUOUS`; no record is deleted and no unknown atom is guessed as donor. Resolved roles are sensitivity-only and cannot be selected after performance inspection.
