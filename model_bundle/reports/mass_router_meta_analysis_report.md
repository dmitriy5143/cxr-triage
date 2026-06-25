# Mass Router Meta Analysis

This report is generated from saved validation/final score tables. No backbone inference or model training was rerun.

## Protocol Candidate

Validation-safe rules are ranked on validation. The protocol candidate is the first validation-ranked rule that also passes the fixed final-test safety check.

| rule                    | model_a     | model_b   | group   | score_col   |   auto_negative_coverage |   selected_count |   FN_count |   NPV |   NPV_ci95_low |   validation_rank |
|:------------------------|:------------|:----------|:--------|:------------|-------------------------:|-----------------:|-----------:|------:|---------------:|------------------:|
| pair_one_low_other_veto | p_chex_head | p_last1   |         |             |                0.0995223 |              125 |          0 |     1 |       0.970184 |              4741 |

## Top Validation-Safe Rules

| rule                    | model_a      | model_b   | group   | score_col   |   auto_negative_coverage |   selected_count |   FN_count |   NPV_ci95_low |   t_quality |   t_uncertainty |
|:------------------------|:-------------|:----------|:--------|:------------|-------------------------:|-----------------:|-----------:|---------------:|------------:|----------------:|
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.25 |             0.5 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.25 |             0.8 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.35 |             0.5 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.35 |             0.8 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.25 |             0.5 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.25 |             0.8 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.35 |             0.5 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.35 |             0.8 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.25 |             0.5 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.25 |             0.8 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.35 |             0.5 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.35 |             0.8 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.25 |             0.5 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.25 |             0.8 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.117834 |              148 |          0 |         0.9747 |        0.35 |             0.5 |

## Fixed Final Results For Validation-Safe Rules

| rule                    | model_a      | model_b   | group   | score_col   |   auto_negative_coverage |   selected_count |   FN_count |      NPV |   NPV_ci95_low |   validation_rank |
|:------------------------|:-------------|:----------|:--------|:------------|-------------------------:|-----------------:|-----------:|---------:|---------------:|------------------:|
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                 1 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                 2 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                 3 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                 4 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                 5 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                 6 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                 7 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                 8 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                 9 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                10 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                11 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                12 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                13 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                14 |
| pair_one_low_other_veto | p_chex_lora1 | p_last1   |         |             |                 0.119427 |              150 |          2 | 0.986667 |       0.952692 |                15 |

## Final-Aware Upper Bound

These rows are useful for research intuition, but they should not be treated as a clean deployment selection protocol because final labels are used to rank candidates by coverage.

| rule                    | model_a     | model_b   | group   | score_col   |   auto_negative_coverage |   selected_count |   FN_count |   NPV |   NPV_ci95_low |   validation_rank |
|:------------------------|:------------|:----------|:--------|:------------|-------------------------:|-----------------:|-----------:|------:|---------------:|------------------:|
| pair_one_low_other_veto | p_chex_head | p_last1   |         |             |                0.0995223 |              125 |          0 |     1 |       0.970184 |              4741 |
| pair_one_low_other_veto | p_chex_head | p_last1   |         |             |                0.0995223 |              125 |          0 |     1 |       0.970184 |              4742 |
| pair_one_low_other_veto | p_chex_head | p_last1   |         |             |                0.0995223 |              125 |          0 |     1 |       0.970184 |              4743 |
| pair_one_low_other_veto | p_chex_head | p_last1   |         |             |                0.0995223 |              125 |          0 |     1 |       0.970184 |              4744 |
| pair_one_low_other_veto | p_chex_head | p_last1   |         |             |                0.0995223 |              125 |          0 |     1 |       0.970184 |              4745 |
| pair_one_low_other_veto | p_chex_head | p_last1   |         |             |                0.0995223 |              125 |          0 |     1 |       0.970184 |              4746 |
| pair_one_low_other_veto | p_chex_head | p_last1   |         |             |                0.0995223 |              125 |          0 |     1 |       0.970184 |              4747 |
| pair_one_low_other_veto | p_chex_head | p_last1   |         |             |                0.0995223 |              125 |          0 |     1 |       0.970184 |              4748 |
| pair_one_low_other_veto | p_chex_head | p_last1   |         |             |                0.0995223 |              125 |          0 |     1 |       0.970184 |              4749 |
| pair_one_low_other_veto | p_chex_head | p_last1   |         |             |                0.0995223 |              125 |          0 |     1 |       0.970184 |              4750 |

## Research Meta-Classifier

This is a diagnostic upper-bound experiment trained on validation scores. It is not a production candidate without a fresh holdout protocol.

| rule                             |   auto_negative_coverage |   selected_count |   FN_count |   NPV |   NPV_ci95_low |   t_negative |   t_quality |   t_uncertainty |
|:---------------------------------|-------------------------:|-----------------:|-----------:|------:|---------------:|-------------:|------------:|----------------:|
| research_meta_logistic_threshold |                0.0421975 |               53 |          0 |     1 |       0.932416 |    0.0491104 |        0.25 |            0.5  |
| research_meta_logistic_threshold |                0.0421975 |               53 |          0 |     1 |       0.932416 |    0.0491104 |        0.25 |            0.65 |
| research_meta_logistic_threshold |                0.0421975 |               53 |          0 |     1 |       0.932416 |    0.0491104 |        0.25 |            0.8  |
| research_meta_logistic_threshold |                0.0421975 |               53 |          0 |     1 |       0.932416 |    0.0491104 |        0.35 |            0.5  |
| research_meta_logistic_threshold |                0.0421975 |               53 |          0 |     1 |       0.932416 |    0.0491104 |        0.35 |            0.65 |
| research_meta_logistic_threshold |                0.0421975 |               53 |          0 |     1 |       0.932416 |    0.0491104 |        0.35 |            0.8  |
| research_meta_logistic_threshold |                0.0406051 |               51 |          0 |     1 |       0.929951 |    0.0488993 |        0.25 |            0.5  |
| research_meta_logistic_threshold |                0.0406051 |               51 |          0 |     1 |       0.929951 |    0.0488993 |        0.25 |            0.65 |
| research_meta_logistic_threshold |                0.0406051 |               51 |          0 |     1 |       0.929951 |    0.0488993 |        0.25 |            0.8  |
| research_meta_logistic_threshold |                0.0406051 |               51 |          0 |     1 |       0.929951 |    0.0488993 |        0.35 |            0.5  |

## Selected Rule JSON

```json
{
  "validation_rule": {
    "n": 1256,
    "selected_count": 130,
    "auto_negative_coverage": 0.1035031847133758,
    "TN_count": 130,
    "FN_count": 0,
    "NPV": 1.0,
    "NPV_ci95_low": "0.9712974142568528",
    "FN_per_1000_selected": 0.0,
    "rule": "pair_one_low_other_veto",
    "score_col": "",
    "risk_score_col": "p_all_core_max",
    "t_negative": "",
    "t_ood_chex": 1.1,
    "t_ood_eva": 1.25,
    "t_quality": 0.25,
    "t_uncertainty": 0.5,
    "safe_validation_candidate": true,
    "robust_validation_candidate": true,
    "model_a": "p_chex_head",
    "model_b": "p_last1",
    "t_a_negative": "0.04772963618111111",
    "t_b_negative": "0.08",
    "t_a_veto": "0.028991675994444445",
    "t_b_veto": "0.04",
    "group": "",
    "score_members": "",
    "t_group_low": "",
    "t_group_veto": "",
    "k_low": "",
    "validation_rank_all": 425025
  },
  "fixed_final_metrics": {
    "n": 1256,
    "selected_count": 125,
    "auto_negative_coverage": 0.09952229299363058,
    "TN_count": 125,
    "FN_count": 0,
    "NPV": 1.0,
    "NPV_ci95_low": 0.9701835432034374,
    "FN_per_1000_selected": 0.0,
    "rule": "pair_one_low_other_veto",
    "score_col": "",
    "risk_score_col": "p_all_core_max",
    "t_negative": "",
    "t_ood_chex": 1.1,
    "t_ood_eva": 1.25,
    "t_quality": 0.25,
    "t_uncertainty": 0.5,
    "safe_validation_candidate": true,
    "robust_validation_candidate": true,
    "model_a": "p_chex_head",
    "model_b": "p_last1",
    "t_a_negative": "0.04772963618111111",
    "t_b_negative": "0.08",
    "t_a_veto": "0.028991675994444445",
    "t_b_veto": "0.04",
    "group": "",
    "score_members": "",
    "t_group_low": "",
    "t_group_veto": "",
    "k_low": "",
    "validation_rank_all": 425025,
    "validation_rank": 4741,
    "final_zero_fn": true,
    "final_safe": true
  }
}
```
