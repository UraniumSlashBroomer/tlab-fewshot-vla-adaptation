# Few-shot адаптация SmolVLA на LIBERO

Репозиторий воспроизводит эксперименты по адаптации
[`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) к трём
held-out задачам LIBERO. Протокол и интерпретация результатов описаны в
[отчёте](report.pdf).

Обучение проводилось на Kaggle/Linux с Tesla P100 и Python 3.12.13. Online
evaluation проводился отдельно на Windows с GeForce GTX 1660 Ti.

## Содержание

- [Установка и данные](#установка-и-данные)
- [Обучение](#обучение)
- [Evaluation на Windows](#evaluation-на-windows)
- [Зафиксированный протокол](#зафиксированный-протокол)
- [Методы адаптации](#методы-адаптации)
- [Результаты](#результаты)

## Установка и данные

Команды ниже приведены для Linux training-машины. Они начинают с чистого
клона репозитория; замените URL на адрес своего fork/репозитория.

```bash
git clone https://github.com/UraniumSlashBroomer/tlab-fewshot-vla-adaptation.git tlab
cd tlab

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout 8f1084e
pip install -e third_party/LIBERO --no-deps --no-build-isolation \
  --config-settings editable_mode=compat

export LIBERO_CONFIG_PATH="$PWD/.libero"
python scripts/setup_libero_config.py
python scripts/download_libero_hdf5.py
```

`download_libero_hdf5.py` скачивает ровно 40 выбранных seen-задач и три
target-задачи из канонического HDF5 source `yifengzhu-hf/LIBERO-datasets`.
Изображения читаются on-the-fly из HDF5; отдельная копия LeRobot dataset не
создаётся. Это отличается от позднее указанного в задании `nvidia/LIBERO_LeRobot_v3`:
в отчёте это отличие раскрыто явно.

Перед дорогим запуском можно проверить, что LIBERO-среда создаётся:

```bash
python scripts/smoke_libero_env.py
```

Для P100 training использовался FP16 и CUDA.

## Обучение

Все команды запускаются из корня репозитория с активированным `.venv`.
Конфиг `configs/train.yaml` уже содержит фактические defaults.

### Seen-претрен

Запустите два независимых seen seed:

```bash
python train.py run.stage=seen run.seed=0
python train.py run.stage=seen run.seed=1
```

Итоговые checkpoints находятся, например, в
`outputs/seen_seed_0/checkpoints/step_010000/policy`.

### Target fine-tune

Baseline для task 0, пяти демонстраций и seed 0:

```bash
python train.py method=baseline run.stage=target run.task_id=0 run.budget=5 run.seed=0 \
  run.init_checkpoint=outputs/seen_seed_0/checkpoints/step_010000/policy
```

Менять метод нужно только Hydra override `method=...`:

```bash
# L2-SP
python train.py method=l2sp run.stage=target run.task_id=0 run.budget=10 run.seed=0 \
  run.init_checkpoint=outputs/seen_seed_0/checkpoints/step_010000/policy

# LoRA
python train.py method=lora run.stage=target run.task_id=0 run.budget=5 run.seed=0 \
  run.init_checkpoint=outputs/seen_seed_0/checkpoints/step_010000/policy
```

Для каждого target budget запускаются task IDs `0, 1, 2` и train seeds `0, 1`.
Методы с небазовой конфигурацией автоматически получают отдельный output path,
например `outputs/lora_expert_target_0_budget_5_seed_0`.

### Resume

Resume восстанавливает сохранённые model, optimizer, scheduler, AMP scaler,
RNG state и исходный training config. Достаточно передать каталог checkpoint:

```bash
python train.py \
  run.resume_checkpoint=outputs/seen_seed_0/checkpoints/step_009000
```

## Evaluation на Windows

Финальные online evaluations выполнялись в Windows PowerShell с WGL. Этот
проект не воспроизводит Linux evaluation: EGL backend для LIBERO на Kaggle
не удалось запустить. Ниже предполагается, что Windows environment уже
подготовлен и PowerShell открыт в корне репозитория. В использованном Windows
`.venv` был Python 3.12.6; он отличается от Python 3.12.13 в Kaggle training
runtime, но сетап был аналогичный, зависимости установились корректно на обоих версиях python.

```powershell
$env:LIBERO_CONFIG_PATH = "$PWD\.libero"
$env:MUJOCO_GL = "wgl"

.\.venv\Scripts\python.exe eval.py `
  checkpoint="outputs\target_0_budget_5_seed_0\checkpoints\step_002000" `
  output_dir="outputs\target_0_budget_5_seed_0_eval" `
  runtime.mujoco_gl=wgl `
  evaluation.task_ids=[0] `
  evaluation.episodes_per_task=20 `
  evaluation.first_rollout_seed=10000 `
  evaluation.policy_sampling_seed=0 `
  evaluation.num_envs=1
```

Вызов оценивает одну target-задачу и пишет `metrics.json` в `output_dir`.
Для полной сетки достаточно повторить команду для каждого task, budget и
train seed.

LoRA checkpoint содержит только adapter weights, поэтому его evaluation
запускается отдельной полной командой с `base_checkpoint`:

```powershell
.\.venv\Scripts\python.exe eval.py `
  checkpoint="outputs\lora_expert_target_0_budget_5_seed_0\checkpoints\step_002000" `
  base_checkpoint="outputs\seen_seed_0\checkpoints\step_010000\policy" `
  output_dir="outputs\lora_expert_target_0_budget_5_seed_0_eval" `
  runtime.mujoco_gl=wgl `
  evaluation.task_ids=[0]
```

## Зафиксированный протокол

| Компонент | Значение |
|---|---|
| Base policy | `lerobot/smolvla_base` |
| Seen data | 40 стратифицированных `libero_90` задач $\times$ 50 demos = 2\,000 эпизодов |
| Target data | `libero_goal`, task IDs 0--2; первые `demo_0...demo_{B-1}` для $B\in\{5,10,25\}$ |
| Trainable baseline modules | Action Expert, action projections и state projection; SmolVLM/VLM заморожен |
| Action interface | 50-action chunk, closed-loop execution одного действия |
| Optimizer | AdamW, lr $10^{-4}$, betas $(0.9,0.95)$ |
| Effective batch | 32: micro-batch 4, gradient accumulation 8 |
| Seen schedule | 10\,000 updates, warmup 1\,000, cosine decay |
| Target schedule | 2\,000 updates, warmup 250, cosine decay |
| Evaluation | 2 train seed, 20 rollout'ов на task--seed, initial-state seeds 10000--10019 |

`train.py` сохраняет `run.json` с data manifest и числом trainable parameters,
`training_config.yaml`, `training_state.pt`, checkpoint policy и `train.log`.
`eval.py` дописывает structured результат в `metrics.json`.

## Методы адаптации

| Hydra method | Изменение относительно baseline | Фиксированные параметры |
|---|---|---|
| `baseline` | Наивный target fine-tune | Без replay и дополнительных loss |
| `l2sp` | L2 regularization от seen checkpoint | $\lambda=10^{-2}$ по trainable parameters |
| `replay` | Seen replay при target fine-tune | 3 target micro-batch : 1 seen micro-batch |
| `lora` | Low-rank adapters вместо обновления base weights | rank 8, $\alpha=16$, dropout 0.05 |

Baseline не использует seen replay на target этапе. Для L2-SP и replay
остаются теми же target demos, optimizer, число updates, preprocessing,
action interface и evaluation protocol. LoRA была намеренно проверена только
на budget 5 как гипотеза против переобучения в наиболее дефицитном режиме.

## Результаты

Числа ниже --- средний success rate по трём target-задачам, двум train seed и
20 rollout'ам на task--seed. Полные task-level tables и qualitative failure
analysis находятся в `research/main.tex`.

| Budget (demos) | Baseline | L2-SP | LoRA |
|---:|---:|---:|---:|
| 5 | 55.0 | 57.5 ($+2.5$) | 74.2 ($+19.2$) |
| 10 | 65.0 | 77.5 ($+12.5$) | --- |
| 25 | 70.0 | 71.7 ($+1.7$) | --- |

L2-SP не показал устойчивого преимущества на всех budgets: основной эффект
получен на task 0 при budget 10 ($37.5\% \rightarrow 67.5\%$). LoRA дала
наиболее сильный наблюдаемый сдвиг при budget 5, особенно на task 0 и task 2,
но не проверялась на budgets 10 и 25. При интерпретации следует учитывать, что
для каждого метода использовались только два независимых training seed.
