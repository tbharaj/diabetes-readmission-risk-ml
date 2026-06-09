# Advanced Evaluation Summary

## Threshold analysis

| threshold | precision | recall | f1 | specificity | predicted_positive_rate | true_positives | false_positives | true_negatives | false_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050 | 0.119 | 0.962 | 0.212 | 0.107 | 0.900 | 2185 | 16140 | 1943 | 86 |
| 0.075 | 0.145 | 0.837 | 0.247 | 0.380 | 0.644 | 1900 | 11216 | 6867 | 371 |
| 0.100 | 0.165 | 0.708 | 0.268 | 0.551 | 0.478 | 1607 | 8118 | 9965 | 664 |
| 0.125 | 0.192 | 0.543 | 0.284 | 0.713 | 0.316 | 1234 | 5195 | 12888 | 1037 |
| 0.150 | 0.218 | 0.397 | 0.281 | 0.821 | 0.203 | 901 | 3234 | 14849 | 1370 |
| 0.175 | 0.238 | 0.313 | 0.270 | 0.874 | 0.147 | 711 | 2281 | 15802 | 1560 |
| 0.200 | 0.269 | 0.244 | 0.256 | 0.917 | 0.101 | 553 | 1503 | 16580 | 1718 |
| 0.250 | 0.350 | 0.122 | 0.180 | 0.972 | 0.039 | 276 | 512 | 17571 | 1995 |
| 0.300 | 0.392 | 0.071 | 0.121 | 0.986 | 0.020 | 162 | 251 | 17832 | 2109 |
| 0.400 | 0.564 | 0.014 | 0.027 | 0.999 | 0.003 | 31 | 24 | 18059 | 2240 |
| 0.500 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0 | 0 | 18083 | 2271 |

## Subgroup/fairness analysis

| group_type | group | n | positive_cases | observed_readmission_rate | predicted_positive_rate | roc_auc | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gender | Female | 10924 | 1255 | 0.115 | 0.326 | 0.685 | 0.196 | 0.557 | 0.290 |
| gender | Male | 9430 | 1016 | 0.108 | 0.304 | 0.676 | 0.186 | 0.527 | 0.275 |
| age | [10-20) | 130 | 6 | 0.046 | 0.069 | 0.843 | 0.333 | 0.500 | 0.400 |
| age | [20-30) | 324 | 43 | 0.133 | 0.290 | 0.832 | 0.351 | 0.767 | 0.482 |
| age | [30-40) | 725 | 78 | 0.108 | 0.241 | 0.710 | 0.211 | 0.474 | 0.292 |
| age | [40-50) | 1913 | 188 | 0.098 | 0.267 | 0.747 | 0.229 | 0.622 | 0.335 |
| age | [50-60) | 3457 | 301 | 0.087 | 0.242 | 0.703 | 0.182 | 0.505 | 0.267 |
| age | [60-70) | 4547 | 529 | 0.116 | 0.292 | 0.657 | 0.195 | 0.490 | 0.279 |
| age | [70-80) | 5234 | 628 | 0.120 | 0.353 | 0.653 | 0.190 | 0.559 | 0.283 |
| age | [80-90) | 3414 | 424 | 0.124 | 0.410 | 0.649 | 0.174 | 0.575 | 0.268 |
| age | [90-100) | 576 | 74 | 0.128 | 0.394 | 0.609 | 0.167 | 0.514 | 0.252 |
| age_band | <60 | 6583 | 616 | 0.094 | 0.247 | 0.731 | 0.210 | 0.555 | 0.305 |
| age_band | >=60 | 13771 | 1655 | 0.120 | 0.349 | 0.653 | 0.186 | 0.539 | 0.276 |
| race | AfricanAmerican | 3866 | 432 | 0.112 | 0.317 | 0.673 | 0.188 | 0.535 | 0.279 |
| race | Asian | 123 | 10 | 0.081 | 0.220 | 0.551 | 0.074 | 0.200 | 0.108 |
| race | Caucasian | 15223 | 1717 | 0.113 | 0.322 | 0.681 | 0.194 | 0.552 | 0.287 |
| race | Hispanic | 404 | 50 | 0.124 | 0.280 | 0.771 | 0.248 | 0.560 | 0.344 |
| race | Other | 276 | 20 | 0.072 | 0.261 | 0.701 | 0.125 | 0.450 | 0.196 |

## Error analysis

| prediction_group | n | mean_predicted_probability | mean_time_in_hospital | mean_num_lab_procedures | mean_num_procedures | mean_num_medications | mean_number_outpatient | mean_number_emergency | mean_number_inpatient | mean_number_diagnoses | most_common_age | most_common_gender | most_common_race | most_common_insulin | most_common_diabetesMed | most_common_discharge_disposition_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| false_negative | 1037 | 0.086 | 4.218 | 43.556 | 1.291 | 15.515 | 0.312 | 0.113 | 0.158 | 7.395 | [70-80) | Female | Caucasian | No | Yes | 1 |
| false_positive | 5195 | 0.181 | 5.366 | 46.310 | 1.238 | 17.969 | 0.550 | 0.386 | 1.629 | 8.090 | [70-80) | Female | Caucasian | No | Yes | 1 |
| true_negative | 12888 | 0.077 | 3.951 | 41.596 | 1.414 | 15.082 | 0.279 | 0.080 | 0.134 | 7.100 | [70-80) | Female | Caucasian | No | Yes | 1 |
| true_positive | 1234 | 0.209 | 5.296 | 45.053 | 1.186 | 18.121 | 0.559 | 0.556 | 2.108 | 8.007 | [70-80) | Female | Caucasian | No | Yes | 1 |

## Interpretation

This advanced evaluation moves the project beyond a simple model comparison by stress-testing the model across validation, calibration, threshold behaviour, subgroup performance and error patterns.
