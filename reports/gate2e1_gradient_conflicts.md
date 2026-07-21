# Gate 2-E1 gradient conflicts

Diagnostics use the first 256 sorted official-train records and frozen seed42 models. They perform no optimizer step and cannot change tasks or weights.

- acceptor_cold/M11: primary-vs-secondary cosine -0.1426; primary-vs-masked None; negative task fraction 63.6%; secondary/primary norm ratio 1.343.
- acceptor_cold/M15: primary-vs-secondary cosine +0.0627; primary-vs-masked 0.042290493845939636; negative task fraction 40.0%; secondary/primary norm ratio 0.585.
- iid/M11: primary-vs-secondary cosine +0.1009; primary-vs-masked None; negative task fraction 27.3%; secondary/primary norm ratio 1.694.
- iid/M15: primary-vs-secondary cosine +0.3246; primary-vs-masked 0.2554945647716522; negative task fraction 0.0%; secondary/primary norm ratio 1.288.

Acceptor-cold M11 shows aggregate negative transfer pressure: primary-vs-secondary cosine is negative and 63.6% of task gradients have negative cosine. This explains uncertainty in transfer but does not authorize reweighting or task removal.
