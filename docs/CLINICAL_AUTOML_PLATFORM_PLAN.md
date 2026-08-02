# Технический план локальной медицинской AutoML-платформы

## 1. Основные объекты

Платформа строится вокруг четырех версионированных объектов:

- `TaskPack` — медицинская задача: формат данных, target, preprocessing, допустимые
  модели, метрики и критерии выпуска;
- `AcquisitionProfile` — производитель и модель прибора, firmware, детектор, протокол,
  разрешение, обработка изображения, клиника и период действия;
- `ModelPack` — неизменяемая версия базовой модели с весами, кодом загрузки, лицензией,
  input contract, требованиями к GPU, контрольными суммами и тестами;
- `SolutionBundle` — готовое решение: `ModelPack + head/adapter + preprocessing +
  calibrator + OOD model + router + validation report`.

PostgreSQL хранит метаданные и состояния. Изображения, snapshots и веса находятся в
локальном S3-compatible хранилище. Все объекты адресуются по версии и SHA-256 digest;
существующий объект никогда не перезаписывается.

## 2. Каталог и выбор backbone-моделей

Запись `ModelPack` содержит:

```text
model_id, version, digest, source, license
supported_tasks, modality, views, input_size, preprocessing_contract
feature_dim, adaptation_methods, VRAM, expected_latency
code_image_digest, SBOM, signature, lifecycle_status
```

Новая модель попадает в каталог только через pipeline импорта:

1. Скачать веса в изолированной среде и зафиксировать источник и лицензию.
2. Проверить контрольные суммы, загрузку без произвольного fallback и один forward pass.
3. Собрать контейнер с закрепленными версиями библиотек и SBOM.
4. Выполнить тесты input/output contract и эталонных примеров.
5. Подписать `ModelPack` и поместить его в локальный registry со статусом `candidate`.

Поиск кандидатов можно частично автоматизировать центральным `Model Scout`: он
отслеживает доверенные репозитории, статьи и новые версии, после чего готовит инженеру
отчет о назначении, лицензии, архитектуре и ожидаемых ресурсах. Scout не может сам
добавить модель в клинический каталог. Инженер подтверждает импорт, фабрика выполняет
проверки и подписывает `ModelPack`, и только после этого пакет становится доступен
локальным агентам клиник.

Для конкретной задачи агент не выбирает модель свободным рассуждением. Registry
сначала детерминированно фильтрует каталог:

```text
совместимая задача и модальность
AND допустимые views/input contract
AND коммерчески допустимая лицензия
AND хватает локального GPU/RAM
AND поддерживается разрешенный метод адаптации
AND пакет подписан и прошел integrity tests
```

Оставшиеся модели ранжируются по локальным evidence: качество на этом типе данных,
устойчивость между приборами, калибровка, скорость и стоимость адаптации. Для нового
профиля сначала запускается дешевый frozen-backbone benchmark. Затем для лучших
кандидатов выполняются head tuning, LoRA и только разрешенная частичная разморозка.

Обновление backbone всегда создает новую версию. Оно инвалидирует старые результаты
совместимости, calibrator и router, после чего проходит полный validation и shadow
deployment. Предыдущая рабочая версия остается доступной для rollback.

## 3. Как агент принимает решения

Агент не имеет прямого доступа к данным и не исполняет код самостоятельно. Через MCP
он читает типизированные resources и вызывает ограниченные tools:

```text
device.inspect             dataset.audit
model.list_compatible      experiment.propose
experiment.start/status    evaluation.compare
calibration.fit            router.optimize
deployment.shadow          deployment.promote/rollback
```

Рабочий процесс:

1. Агент получает `PlatformState`: задачу, профиль прибора, snapshots, текущую модель,
   доступное железо, историю запусков и ограничения.
2. Агент формирует только структурированный `ExperimentPlan` JSON: кандидаты, методы
   адаптации, search space, budget, splits и критерии остановки.
3. Детерминированный Policy Engine проверяет план по `TaskPack`: лицензии, ресурсы,
   запреты, отсутствие test leakage и допустимые действия.
4. Workflow Engine исполняет одобренный DAG в контейнерах. Агент может наблюдать и
   объяснять прогресс, но не менять вычисленные метрики.
5. Evaluator формирует `EvaluationReport`. Правила выбора применяются кодом, а не LLM.
6. Агент объясняет результат и предлагает действие. Выпуск модели требует approval.

Сначала применяется safety gate: целостность данных, минимальное число событий,
доверительные интервалы, отсутствие регрессии, корректная калибровка, latency и
стабильность подгрупп. Прошедшие кандидаты ранжируются лексикографически:
AUROC, AUPRC, Brier/ECE, безопасная доля автоматизации, затем latency. Для deployment
используются отдельно зафиксированные требования по NPV/FN. Агент не может их ослабить.

## 4. Обучение, калибровка и validation

Каждый запуск использует immutable `DatasetSnapshot` и patient-level split:

```text
train -> calibration -> validation -> locked final test
```

Snapshot содержит список исследований, target provenance, профиль прибора, проверки
дубликатов и hash каждого входа. Final test не используется для выбора модели,
calibrator, thresholds или router.

Для каждого кандидата workflow выполняет:

1. data QA и проверку target;
2. обучение с checkpointing, early stopping и журналом параметров;
3. выбор calibration method только на calibration split;
4. поиск router по validation: negative/positive thresholds, quality, OOD,
   uncertainty и gray zone;
5. однократную оценку зафиксированной конфигурации на final test;
6. bootstrap confidence intervals, subgroup и device-profile analysis;
7. latency/memory benchmark и сравнение с действующим champion;
8. генерацию model card, error report и списка случаев для врачебного review.

В текущем MVP automatic/scheduled retraining юридически отключен. Описанный ниже
контур является будущей архитектурой и может быть включен только после отдельного
правового и клинического утверждения. Scheduled retraining сначала проверяет триггеры: достаточное число новых подтвержденных
labels, data drift, изменение оборудования или ухудшение мониторинга. Без триггера
обучение не запускается. Feedback включает случайный аудит auto-negative случаев,
чтобы не возникал verification bias.

## 5. Сборка и выпуск решения

Прошедший кандидат собирается в:

```text
solution_bundle/
  manifest.json
  model_pack_ref.json
  head_or_adapter/
  preprocessing.json
  calibrator/
  ood_model/
  router.json
  validation_report.json
  model_card.md
  environment.lock
  sbom.json
  checksums.sha256
  signature/
```

Build pipeline проверяет hashes, выполняет fresh-environment smoke test, replay
эталонных случаев и parity test с research output. Затем bundle подписывается и
регистрируется как `validated`.

Статусы выпуска:

```text
candidate -> validated -> shadow -> active -> retired
```

В shadow-режиме новая версия получает реальные входы, но ее ответы не влияют на маршрут
врача. После заданного периода Evaluator сравнивает champion и challenger. `Promote`
разрешается только пользователю с соответствующей ролью; deployment хранит digest
активного bundle и digest rollback-версии.

## 6. Безопасность и проверка агента

MCP-серверы не предоставляют shell, произвольный SQL, общие файловые операции или
отправку на внешний URL. Каждый вызов имеет scope, resource limit, dry-run,
idempotency key и audit record. Медицинские данные, отчеты и embeddings остаются
локально; исходящая сеть запрещена по умолчанию.

Перед выпуском агента выполняются:

- contract-тесты всех MCP tools и проверка прав;
- сценарии разрешенных и запрещенных действий;
- prompt-injection тесты через DICOM metadata и клинический текст;
- повтор одного `PlatformState` с проверкой одинакового исполнимого плана;
- сбои worker/storage/network и безопасное продолжение или rollback;
- проверка, что агент не может выпустить модель, изменить safety gate или экспортировать
  данные без approval;
- полный audit replay: кто, когда, на каких данных и почему создал deployment.

Первый технический релиз готов, когда текущий CXR/ФЛГ-контур упакован как `TaskPack`,
две версии backbone проходят описанный lifecycle, агент воспроизводимо создает план,
а собранный `SolutionBundle` повторяет исследовательские метрики без дрейфа.
