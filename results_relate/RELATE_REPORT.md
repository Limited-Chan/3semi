# Step 2: relationships

Representative statistic explicitly chosen: **median**.

Recorded reason: **掲載日数の分布は長期側に裾があり、平均値が中央値より大きく、最頻値0日も代表値として適さないため、典型的な掲載日数の比較には中央値を用いる**

## Order of analysis

1. `01_one_to_one/` — one attribute at a time.
2. `02_multiple_variables/money_room_by_house_area_...png` — then increase to two attributes.
3. `02_multiple_variables/age_by_layout_table.csv` — age × layout is kept as a table because layout is nominal; lines would falsely imply layout order.

## What is deliberately not done

- No causal interpretation.
- No p-values or independence assumption.
- No automatic outlier deletion.
- No heatmap or boxplot as a primary requirement output.
- No top-coding/overflow grouping; display paging never changes the underlying CSV statistics.
- No automatic property-type exclusion unless it was explicitly specified in Step 1.
