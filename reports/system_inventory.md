# System inventory

Observed at 2026-07-18 13:43 UTC.

- CPU: 2 × AMD EPYC 7763, 64 cores/socket, SMT2; 256 logical CPUs.
- NUMA: 2 nodes.
- RAM: 440 GiB total, 48 GiB used, 349 GiB free, 391 GiB available.
- Swap: 31 GiB total, 0 used.
- Filesystem: `/dev/nvme0n1p4`, 3.5 TiB total, 2.7 TiB used, 596 GiB available, 83% used.
- `/home/changliu/ExcitationNexus_Data_v2`: 3.4 GiB allocated by `du -sh`.
- GPU: seven RTX 3090 cards, each 24,576 MiB. All met the Gate 0-A FREE rule at observation time.

Disk capacity is adequate for manifests, smoke tests, and checkpoints, but 83% filesystem utilization requires per-run retention discipline.
