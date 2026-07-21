# Gate 2-D2 v2 sequence-length correction

This target-free amendment was frozen before any molecular forward, embedding, or validation result. The v1 audit counted tokens with a regex; the immutable model tokenizer gives maxima 417/208/378 for full/donor/acceptor inputs rather than 399/208/372. The hard gate is strengthened to test the true maxima. No model, molecular string, projection, XGBoost contract, bootstrap rule, or admission threshold changed.
