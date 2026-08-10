# Зафиксированные решения эксперимента

Дата фиксации: 2026-08-08.

## Данные и split

- Seen-pretraining: канонический suite `libero_90`, содержащий ровно 90 задач.
- Подмножество seen-задач: фиксированный стратифицированный набор из `40` task IDs:
  - kitchen (`20`): `0, 2, 6, 9, 11, 16, 18, 20, 22, 27, 28, 32, 33, 34, 35, 37, 38, 39, 42, 45`;
  - living room (`13`): `46, 48, 50, 52, 55, 57, 60, 61, 63, 65, 68, 69, 71`;
  - study (`7`): `74, 77, 78, 81, 83, 87, 89`.
  Внутри каждой сцены набор покрывает разные объекты, receptacle и примитивы (open/close, transfer, relative placement, stacking, switch). Выбор не использует target-демонстрации или результаты evaluation и не будет меняться по результатам.
- Target tasks: первые три задачи `libero_goal` (task IDs `0--2`), как требует задание.
- Для target-adaptation разрешены только первые по числовому source ID `5`, `10` или `25` демонстраций соответствующей target-задачи: `demo_0..demo_4`, `demo_0..demo_9` или `demo_0..demo_24`. HDF5 keys не используются в лексикографическом порядке. Демонстрации других target-задач не используются.
- Демонстрации seen-задач разрешены только как явно описанный replay/mix в нашем методе; baseline этого не использует.
- Точка `0` не является отдельной целью оптимизации: baseline и все методы используют одни и те же два seen-checkpoint'а и поэтому имеют идентичный zero-shot результат. Мы не запускаем отдельный seen-pretrain с альтернативной конфигурацией ради улучшения `0`.

## Границы методов

- Baseline сохраняет нативную архитектуру и action expert SmolVLA: continuous action chunks, conditional flow matching и исходный action interface.
- Для baseline фиксируются нативные preprocessing transforms LeRobot/SmolVLA: выбор и порядок камер, resize, нормализация изображений, токенизация текста и нормализация actions. Они применяются одинаково в baseline и во всех методах.
- Дополнительные stochastic train-time image augmentations не входят в baseline. Они могут быть отдельной гипотезой метода и тогда применяются только на обучающих кадрах; evaluation observations никогда не аугментируются.
- Мы не тестируем замену action head, замену flow-matching objective или крупную архитектурную переделку action expert: это выходит за доступный compute budget.
- Допустимые направления улучшений поверх baseline: scope trainable parameters (например, expert-only / LoRA), replay/mix из seen-демонстраций, умеренные train-time аугментации и настройки native chunk/replanning после проверки совместимости с evaluation pipeline.

## Основная метрика

Success rate на `libero_goal` task IDs `0--2` при budgets `0/5/10/25`, с итогом как среднее по трём задачам. Для финального сравнения используются два train seeds (`0`, `1`) и ровно 40 online evaluation episodes на task-точку с initial-state seeds `10000..10039`; один и тот же набор применяется ко всем методам и baseline.

## Training schedule baseline

- Seen-pretrain: `30,000` optimizer updates, effective batch size `32`, linear warmup `1,000` updates followed by cosine decay. Checkpoint every `10,000` updates, retaining two latest checkpoints.
- Target fine-tune: `5,000` optimizer updates for every budget, effective batch size `32`, linear warmup `250` updates followed by cosine decay. Checkpoint every `2,500` updates, retaining two latest checkpoints.
- Initial configuration uses micro-batch `4` and gradient accumulation `8`. A GPU smoke run verifies memory and timing before any full run; if micro-batch changes, accumulation changes proportionally to keep effective batch size fixed.

## Обоснование

40 seen-задач — достаточный первый бюджет, чтобы покрыть разнообразные короткие manipulation skills и сохранить compute для честной кривой baseline, двух сидов и абляций. В отличие от первых 40 task IDs (которые почти полностью принадлежат kitchen), выбранный набор покрывает все три домена канонического `libero_90`. Полный `libero_90` остаётся возможным последующим масштабированием, но не будет подменять этот зафиксированный baseline.
