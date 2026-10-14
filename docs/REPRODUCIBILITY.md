# Numerical portability

The release retains exact rankings and predictions from the recorded experiment. Metrics, calibration choice, claim/reference boundaries, paired corruption cases and review status are recomputed exactly from those ledgers.

Linux CI exposed a brittle check: it required every regenerated similarity ranking to have identical order to the Apple CPU run, including numerically tied or almost tied scores. The portable replay now verifies that every released lexical/dense/projection ranking is descending and a valid top-k selection within an **absolute similarity tolerance of 1e-6**. It rejects inversions or excluded higher-scoring documents beyond that tolerance. It logs how many regenerated orders differ and reconstructs the hybrid ranking exactly from the released component rankings.

This does not assert bitwise-identical rankings across BLAS implementations. Near a retrieval cutoff, numerical ties can alter identity-based recall; the primary reported numbers use the explicitly released ranking ledger. This change did not retrain models, change labels, modify released rankings, tune a retrieval parameter against development accuracy, or manufacture reviewer judgments.
