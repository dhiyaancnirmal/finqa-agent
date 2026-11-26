# Example episode: question 2 (super-120b__harness)

Question: For company `alphabet`, here is the question : What is the ratio of Domestic Income to Foreign Income for continuing operations before income taxes in 2022?

Ground truth: `\boxed{6.118}`. Submitted: `6.118`. Reward: 1.0.

**1. `get_descriptions`** `{}`

```
["us_gaap_InvestmentsClassifiedByContractualMaturityDateTableTextBlock", "us_gaap_ScheduleOfIncomeBeforeIncomeTaxDomesticAndForeignTableTextBlock", "us_gaap_LeaseCostTableTextBlock", "us_gaap_LesseeOperatingLeaseLiabilityMaturityTableTextBlock", "us_gaap_DisaggregationOfRevenueTableTextBlock", "us_gaap_ScheduleOfUnrecognizedTaxBenefitsRollForwardTableTextBlock", "us_gaap_ScheduleOfEarningsPerShareBasicAndDilutedTableTextBlock", "goog_AssetsAndLiabilitiesLesseeTableTextBlock", "us_gaap_ScheduleOfEffectiveIncomeTaxRateReconciliationTableTextBlock", "us_gaap_DebtSecuritiesAvailableForSaleTableTex ...
```

**2. `get_table_info`** `{"table_name": "us_gaap_ScheduleOfIncomeBeforeIncomeTaxDomesticAndForeignTableTextBlock"}`

```
{"description": "income_from_continuing_operations_before_income_taxes_consisted_of_the_following_(in_millions):","column_names": ["operations","2022","2023","2024"],"unique_vals_per_col": {"operations": ["domestic_operations","foreign_operations","total"]},"company": "alphabet","column_dtypes": {"operations": "str","2022": "str","2023": "str","2024": "str"}}
```

**3. `sql_query`** `{"table_name": "us_gaap_ScheduleOfIncomeBeforeIncomeTaxDomesticAndForeignTableTextBlock", "query": "SELECT \"operations\", \"2022\" FROM us_gaap_ScheduleOfIncomeBeforeIncomeTaxDomesticAndForeignTableTextBlock WHERE \"operations\" IN ('domestic_operations', 'foreign_operations')"}`

```
[{"operations":"domestic_operations","2022":"$61,307"},{"operations":"foreign_operations","2022":"$10,021"}]
```

**4. `calculate`** `{"expression": "61307 / 10021"}`

```
6.1178525097
```

**5. `submit_answer`** `{"answer": "6.118"}`

```
Before this answer is final, check it:
- Does it answer exactly what was asked (right line item, right year, right direction of the ratio or change)?
- Are percentages and growth rates written as decimal ratios (22% -> 0.22)?
- Are the values in the table's units, not multiplied out?
- Was every number computed with the calculate tool from values you actually queried?
If anything is wrong, fix it with more tool calls. Then call submit_answer again with the final answer only (it will be recorded as: 6.118).
```

**6. `submit_answer`** `{"answer": "6.118", "raw": "6.118"}`

```
Answer submitted: 6.118
```
