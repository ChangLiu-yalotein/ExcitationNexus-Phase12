# Gate 1-B3 target-free radius edge cache

Coverage is **15,016/15,016** with zero missing/duplicate IDs. A deterministic stratified sample of **256** records had **0** edge and **0** atom-order mismatches. Dynamic construction processed 6.04 graphs/s; cached lookup processed 172.44 graphs/s (28.55x). The 30 local shards occupy 99.4 MiB. The cache contains only identities, atom counts, radius edges, and source graph hashes; distances are reconstructed from frozen DFT coordinates. No target scalar was read. GPU waiting time during this CPU benchmark was 0 s.
