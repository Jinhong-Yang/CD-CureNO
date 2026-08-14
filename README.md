# CD-CureNO

CD-CureNO is a Python research-software stack for auditable neural-operator
development in thermochemical composite curing. It provides conservative
one- and two-dimensional reference solvers, immutable data manifests,
joint-field and causal operators, restriction-preserving checkpoint inflation,
managed execution, and independent evaluation/reporting utilities.

## Scientific scope

The public software claim is limited to functionality validated in stages
P0--P5. The unfinished P6 held-out campaign is not used to claim predictive
superiority, broad generalization, or experimental validation. The included
SoftwareX manuscript records the numerical evidence and its claim boundaries.

## Installation

CD-CureNO supports Python 3.10--3.12.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

CPU execution is sufficient for audits and tests. CUDA is optional for model
training.

## Repository layout

- `src/cdcureno/`: installable Python package.
- `scripts/`: auditable command-line workflows.
- `configs/`: frozen experiment and solver configurations.
- `tests/`: regression, contract, and scientific-integrity tests.
- `analysis/`: validation records and claim-boundary documents.
- `manuscript/softwarex_overleaf/`: SoftwareX manuscript source and evidence
  traceability.

## Data and upstream software

Generated arrays and third-party datasets are not implicitly relicensed by
this repository. The upstream ResFNO code/data have no explicit license
recorded in this project and must not be redistributed as CD-CureNO content.
Use only documented acquisition steps, checksums, and upstream terms.

## Authors and support

- Hyojin Park, Gyeongnam Intelligence Innovation Center, Kyungnam University.
- Nam-Hyun Yoo, Computer Engineering, Kyungnam University.
- Jinhong Yang, Department of Medical Information Technology, Inje University.

Correspondence and support: `jinhong@inje.ac.kr`.

## Citation

Citation metadata are provided in `CITATION.cff`. Cite the tagged release used
in an analysis and, once published, the accompanying SoftwareX article.

## License

CD-CureNO is open-source software released under the
[Apache License 2.0](LICENSE.txt) (`Apache-2.0`). The license permits use,
modification, and redistribution, including commercial use, subject to its
terms. Third-party code and data retain their own licenses and are not covered
by the CD-CureNO license merely because they are referenced by this project.
