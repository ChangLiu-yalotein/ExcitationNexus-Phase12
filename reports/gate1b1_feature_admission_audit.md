# Gate 1-B1 feature admission audit

All 15,016 records joined one-to-one. The target-free cache contains 532 C0 features and 535 C1.5-safe features. Cache SHA-256: `7bb1a6af207364b186c072ab6d013d278e294633255b2235ef789e040e75d058`.

The old 541-column no-dipole model cannot be migrated verbatim: its PM6 energy semantics are unresolved, and atom count, termination, warning, missingness-control, and duplicate-unit gap fields are outside the new safe contract. Only HOMO, LUMO, and gap_eV are admitted. PM6 dipole and all DFT/TDDFT fields are excluded.

RDKit `2025.09.6`; Morgan radius 2, 512 bits, includeChirality=False. No descriptor parse or missing-value failure occurred.
