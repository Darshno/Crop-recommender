# Unresolved data questions

These questions must be answered by the data owners before retraining:

1. What is the authoritative source, version, and acquisition date for each CSV?
2. Are N, P, and K measured in the same units, soil basis, depth, and laboratory method?
3. Is rainfall a monthly total, seasonal total, or another aggregation window?
4. What aggregation windows and instruments produced temperature and humidity?
5. How were pH values sampled and measured?
6. Does `label` mean observed planted crop, agronomist recommendation, optimal crop,
   yield winner, or another target?
7. Are independent labels expert-verified, and are outcomes such as yield or crop
   failure available?
8. Are rows synthetic, augmented, simulated, or directly observed?
9. Are there farm/field/site/time identifiers that must remain grouped?
10. Can the independent rows be proven independent from the original data and from
    any data used to construct the existing models?
