# Data

## `stability_constants_dative_consolidated.csv`

19,964 unique metal-ligand complexes spanning 61 metals, all measured at 25 °C.

| Column | Meaning |
|---|---|
| `smiles` | Dative-bond SMILES; `->` / `<-` mark METAL_COORD bonds |
| `logK1` | First stability constant, log10 units (median across sources) |
| `metal_type` | Element symbol; oxidation state is carried in the SMILES |
| `n_atoms` | Heavy-atom count |
| `n_dative_bonds` | Number of dative bonds (1 or 2) |
| `donor_elements` | Donor element signature, e.g. `N`, `N+O` |
| `n_measurements` | Number of source measurements merged into this row |

log K1 ranges from −1.91 to 30.70, mean 6.65. The five most abundant metals are
Cu (2,683), Ni (1,805), Zn (1,686), Co (1,351) and Fe (734).

## `example_ligands.csv`

The curated 99-ligand benchmark used for coordination probing.

## Sources and terms

| Source | Contribution |
|---|---|---|
| IUPAC Stability Constants Database, via Karunaratne et al. 2025 (JCIM) | 32,459 entries pre-curation | 
| LOGKPREDICT, Zahariev et al. 2024 | IUPAC (5,096) and NIST (1,662) subsets | 
| NIST Standard Reference Database 46 | 8,999 records resolved, 7,404 converted |
| Manual curation | 176 Ga/In/Tl/Be entries | This work |
| PubChem PUG-REST | Name-to-SMILES resolution | NCBI usage policy; |

The **Toney coordination benchmark is not redistributed here.** Obtain it from
Zenodo record 13840776 and place it under `data/toney/`; 

