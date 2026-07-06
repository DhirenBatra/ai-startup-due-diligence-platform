# Data Dictionary — Final Feature Set

Total columns: 38

| Column | Type | Description |
|---|---|---|
| state_code | str | Raw feature from source dataset |
| labels | int64 | Target variable: 1 = acquired/success, 0 = closed/failure |
| age_first_funding_year | float64 | Raw feature from source dataset |
| age_last_funding_year | float64 | Raw feature from source dataset |
| age_first_milestone_year | float64 | Raw feature from source dataset |
| age_last_milestone_year | float64 | Raw feature from source dataset |
| relationships | int64 | Number of business relationships/connections |
| funding_rounds | int64 | Number of funding rounds |
| funding_total_usd | int64 | Total funding raised (USD) |
| milestones | int64 | Number of milestones achieved |
| is_CA | int64 | Raw feature from source dataset |
| is_NY | int64 | Raw feature from source dataset |
| is_MA | int64 | Raw feature from source dataset |
| is_TX | int64 | Raw feature from source dataset |
| is_otherstate | int64 | Raw feature from source dataset |
| is_software | int64 | Raw feature from source dataset |
| is_web | int64 | Raw feature from source dataset |
| is_mobile | int64 | Raw feature from source dataset |
| is_enterprise | int64 | Raw feature from source dataset |
| is_advertising | int64 | Raw feature from source dataset |
| is_gamesvideo | int64 | Raw feature from source dataset |
| is_ecommerce | int64 | Raw feature from source dataset |
| is_biotech | int64 | Raw feature from source dataset |
| is_consulting | int64 | Raw feature from source dataset |
| is_othercategory | int64 | Raw feature from source dataset |
| has_VC | int64 | Raw feature from source dataset |
| has_angel | int64 | Raw feature from source dataset |
| has_roundA | int64 | Raw feature from source dataset |
| has_roundB | int64 | Raw feature from source dataset |
| has_roundC | int64 | Raw feature from source dataset |
| has_roundD | int64 | Raw feature from source dataset |
| avg_participants | float64 | Raw feature from source dataset |
| is_top500 | int64 | Whether the startup is in a top-500 index |
| funding_velocity | float64 | Engineered: funding raised per year of funding activity |
| funding_per_round | float64 | Engineered: average funding amount per round |
| milestone_rate | float64 | Engineered: milestones achieved per year |
| regulatory_risk_flag | int64 | Engineered: 1 if in a heavily regulated sector (biotech) |
| churn_risk_flag | int64 | Engineered: 1 if in a consumer-facing, churn-prone sector |
