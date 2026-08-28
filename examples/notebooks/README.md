# Notebooks

The same four worked examples as [../](..), as executed Jupyter notebooks rather than
scripts — self-contained, no dependency on the sibling `.py` files, real (already-run)
output cells so you can read the results on GitHub without a kernel.

- **adult_census.ipynb** — the core argument: synthkit vs. Faker-style shuffling vs. real
  correlation, on UCI Adult Census Income.
- **titanic.ipynb** — null co-occurrence, free text, identifiers, a small dataset.
- **wine_quality.ipynb** — full correlation-matrix fidelity across 12 mostly-continuous
  columns.
- **bike_sharing.ipynb** — a real datetime column, and a real derived-column relationship
  (`cnt = casual + registered`) demonstrating the `Derived` constraint.

To re-run one yourself: `pip install -e ..[dev] jupyter` from this directory's parent, then
`jupyter notebook adult_census.ipynb` (or `jupyter nbconvert --to notebook --execute
--inplace *.ipynb` to refresh all four in place). Downloaded datasets are cached in
`../data/` (gitignored), shared with the script versions.
