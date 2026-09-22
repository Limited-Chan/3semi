$ErrorActionPreference = "Stop"
.\.venv\Scripts\python.exe analyze.py inspect `
  --input "prepared\kokubunji.csv.gz" `
  --out "results_inspect" `
  --years "2020,2021,2022,2023" `
  --types all `
  --duplicates keep `
  --rent-step-yen 20000 `
  --rent-bins-per-figure 10 `
  --area-step-sqm 10 `
  --area-bins-per-figure 10 `
  --age-step-years 5 `
  --age-bins-per-figure 10 `
  --duration-step-days 30 `
  --duration-bins-per-figure 12
