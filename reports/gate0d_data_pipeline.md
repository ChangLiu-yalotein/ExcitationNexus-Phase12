# Gate 0-D unified data pipeline

The reusable package is `src/excitationnexus_phase12`. It binds the frozen table and six manifests by SHA-256 and performs a one-to-one `molecule_id` join independent of source-table row order.

Supported views are `table_only`, `tier0_2d`, `tier1_pm6_3d`, and `tier2_dft_3d`. The main smoke uses PM6 geometry with PM6 HOMO, LUMO, and gap; PM6 dipole is disabled. DFT is loader-only upper-bound compatibility in Gate 0-D. Raw paths locate files but are never features.

Loaders accept only explicit `train`, `val`, or `test`. `buffer` and `historical_quarantine` raise immediately and cannot enter a loader through complement filtering.

Graph construction reads actual atomic numbers, coordinates, and JSON atom roles and requires exact equality with sidecar `atom_origins`. Directed cutoff edges use 5.0 Å and at most 32 nearest neighbors per destination.

## Role-label limitation discovered

Full read-only role counting found identical PM6/DFT distributions: 14,263 records contain donor/acceptor only; 366 also contain explicit unknown atoms; 387 have no donor-labelled atom and contain explicit unknown atoms; no record lacks acceptor-labelled atoms. Unknown is not inferred as donor. The tiny plumbing model pools unknown atoms separately and uses donor/acceptor presence flags; an empty donor pool is a zero vector with presence=0. Formal models must preserve this missing-role information or resolve it through a separately audited annotation correction.

Six complete joins and 72 on-demand PM6/DFT graph parses passed. Graphs are not permanently materialized in memory.
