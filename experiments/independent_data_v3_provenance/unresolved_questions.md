# Unresolved provenance and compatibility questions

## Local files

1. What exact release, URL, checksum, and acquisition date produced each CSV?
2. Are the original and independent files derived from the same upstream source?
3. Are N/P/K reported in mg/kg, ppm, kg/ha, or another unit?
4. What soil depth, extraction chemistry, laboratory method, and moisture basis
   apply to each nutrient?
5. What does rainfall represent: daily, monthly, seasonal, or growing-season
   accumulation?
6. What are the temperature and humidity source, spatial resolution, and
   aggregation window?
7. How was pH sampled and measured?
8. Does `label` mean observed crop, recommended crop, optimal crop, or another
   target?
9. Were labels expert-reviewed? Are yield, failure, profit, or management
   outcomes available?
10. Are rows observed, simulated, synthetic, augmented, or copied from another
    dataset?
11. Are there field/farm/site IDs, coordinates, dates, seasons, or source IDs?
12. Can duplicate or related observations be grouped before evaluation?

## Candidate-source acquisition

1. Which exact Soil Health Card/OGD release is licensed and downloadable?
2. Does that release contain row-level soil results rather than aggregate
   summaries?
3. Can NASA POWER or another weather source be linked to the same field and
   time period as soil observations?
4. Can FAOSTAT outcomes be linked to the same geography and crop labels without
   treating production as recommendation ground truth?
5. Which ISRIC SoilGrids depth layer and property definitions would be valid for
   the intended soil measurements?
6. What license, attribution, API limits, and redistribution rules apply to the
   selected releases?
