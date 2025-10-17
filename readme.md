# ATML-PA02

## Running the Experiments
Execute the scripts from the `Task1` directory so the relative data paths resolve:

```powershell
cd Task1
python a.py   # Source-only baseline
python b.py   # Domain-alignment methods (DAN, DANN, CDAN)
python c.py   # Self-training with pseudo-labelling
python d.py   # Concept-shift and rare-class scenarios
```

## Script Guide (One-Liners)
- `a.py`: trains a source-only ResNet-50 baseline and reports cross-domain accuracy gaps with diagnostic plots.
- `b.py`: compares three domain-alignment techniques (DAN, DANN, CDAN) including proxy A-distance analysis and rare-class metrics.
- `c.py`: bootstraps target performance via self-training, iteratively adding confident pseudo-labels on the target domain.
- `d.py`: stress-tests the baseline, DAN, and DANN under concept shift and rare-class scarcity, highlighting per-class robustness.
