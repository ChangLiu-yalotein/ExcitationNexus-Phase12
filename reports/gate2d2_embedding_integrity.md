# Gate 2-D2 embedding integrity

Status: **NOT_EXTRACTED_DUE_TO_PCA_CONTRACT_BLOCKER**.

The pinned model and tokenizer passed security/integrity admission, but no remote code or weight was loaded and no embedding was extracted because the downstream preregistered PCA arm is mathematically infeasible. This preserves the rule that expensive assets are not executed after an earlier fail-closed gate.
