# Mass Router Meta Analysis: краткий вывод

## Что было сделано

Проведен массовый posthoc-прогон по уже сохраненным score-таблицам без повторного обучения backbone-моделей. В переборе участвовали сигналы:

- `p_last1`, `p_last2`: EVA-X-B partial unfreeze;
- `p_chex_head`: tuned MLP head поверх frozen CheXFound features;
- `p_chex_frozen`: исходная frozen CheXFound ветка;
- `p_chex_lora1`, `p_chex_lora2`: два CheXFound LoRA прогона;
- quality/OOD/uncertainty gates.

Проверялись одиночные пороги, pair-veto правила, group-consensus правила и диагностический research-only meta-classifier.

## Лучший deployment-style router

Лучшим финально безопасным правилом стал ансамбль:

- CheXFound tuned frozen head: `p_chex_head`;
- EVA-X-B partial unfreeze last1: `p_last1`;
- правило: `pair_one_low_other_veto`.

Auto-negative выставляется только если выполняется одно из условий:

- `p_chex_head <= 0.0477296` и `p_last1 <= 0.04`;
- или `p_last1 <= 0.08` и `p_chex_head <= 0.0289917`.

Дополнительные фильтры:

- `ood_score_chex <= 1.10`;
- `ood_score_eva <= 1.25`;
- `quality_score >= 0.25`;
- max uncertainty по двум score `<= 0.50`.

## Итоговые метрики на final test

| metric | value |
|---|---:|
| final test size | 1256 |
| auto-negative cases | 125 |
| auto-negative coverage | 9.95% |
| FN inside auto-negative route | 0 |
| NPV | 1.000 |
| NPV CI95 low | 0.9702 |
| FN per 1000 selected | 0.0 |

Относительно предыдущего лучшего router-кандидата это дает +11 auto-negative случаев на final test: было 114/1256, стало 125/1256.

## Почему не берем более агрессивные LoRA-правила

Самые высокие validation-кандидаты с `p_chex_lora1 + p_last1` доходили до 11.78% auto-negative на validation без FN, но на fixed final test давали 2 FN. Поэтому они отвергнуты safety gate и не должны идти в MVP как deployment router.

Это важный результат: CheXFound LoRA полезна как исследовательский сигнал, но текущая aggressive LoRA-veto политика неустойчива на final.

## Что мешает дальнейшему росту auto-negative

На final test у выбранного правила нормальные снимки чаще всего не проходят в auto-negative по двум причинам:

- score/veto еще недостаточно низкие: 305 normal cases;
- высокая uncertainty: 194 normal cases.

Есть также несколько positive near-boundary cases, очень близких к расширению auto-negative. Это объясняет, почему простое ослабление порогов быстро ломает FN=0.

## Методологический вывод

Posthoc router/threshold ресурс почти исчерпан: массовый sweep поднял coverage с 9.08% до 9.95%, но дальнейшее безопасное расширение упирается не в отсутствие порогового правила, а в качество разделения низкорисковых снимков на уровне model scores.

Следующий осмысленный шаг для роста автоматизации: улучшать score separation через новый адаптер/дообучение, возможно с использованием bbox-aware интерпретации VinDr как sanity/auxiliary источника, а затем заново калибровать router.

## Основные файлы

- `mass_router_meta_analysis_report.md`: полный технический отчет;
- `selected_mass_router_config.json`: выбранное правило и параметры;
- `selected_routes_final_test.csv`: case-level маршруты на final test;
- `final_safe_from_validation_candidates.csv`: все validation-кандидаты, прошедшие fixed final safety;
- `router_blocker_summary_final_test.csv`: причины, почему случаи не попали в auto-negative;
- `positive_boundary_risk_cases_final_test.csv`: positive near-boundary cases;
- `normal_blocked_near_boundary_cases_final_test.csv`: normal cases, которые ближе всего к расширению auto-negative.
