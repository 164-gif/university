# Этап 7.1 — полный эксперимент

50 контрастивных эпох (все базовые запросы в каждой эпохе), перед ними 5 эпох MLM. Словарь BPE обучается с нуля на Train, включая удаление точных совпадений с heldout. Это устраняет известную утечку полного словаря; смысловые дубли всё ещё возможны. Старый pilot остаётся неизменным.

Расширение запросов: до 16 вариантов на запрос: контролируемые преобразования формулировок определений и тем, а также стилевые префиксы; исходная формулировка сохраняется. Дубликаты и неоднозначные варианты исключаются. Цель 30–50 тыс. вариантов, фактическое число в augmentation-manifest.json. Это НЕ 30–50 тыс. независимых смысловых перефразировок; основную долю составляют шаблоны. В каждой из 50 эпох все базовые запросы посещаются один раз с ротацией вариантов; обучение не делает 50 полных проходов по каждому расширенному варианту.

MLM: маскируются 15% обычных токенов (80% MASK, 10% случайный, 10% исходный), общая с encoder матрица эмбеддингов используется как выходная проекция. Предобучение только на train passages.

Семантическая дистилляция: intfloat/multilingual-e5-small, revision fd1525a9fd15316a2d503bf26ab031a61d056e98, MIT. Учитель заморожен, использует query:/passage: и masked mean pooling. Его векторы базовых train вопросов и фрагментов вычисляются один раз, затем используются для KL по кандидатам и смягчения вероятных false negatives. Варианты запроса наследуют оценку базового вопроса. Учитель не нужен при браузерном inference. Используются только safetensors, trust_remote_code=False. Учебный текст обрабатывается на runner, не передаётся в API inference.

Validation выбирает best checkpoint и RRF. Heldout Test и отдельные вопросы из paraphrases_test.json используются только в конце. Вопросы составлены ассистентом по источникам, а не независимым человеком; набор мал, разметка требует экспертной проверки. Отсутствующие anchors перечисляются. Не применять результаты как объективную оценку всех дисциплин. Не квантовать слабые веса до проверки.

```bash
python -m pip install -r ai/model/requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r ai/training/requirements-full.txt
python -m ai.indexing.build_index
python -m ai.training.prepare_full
python -m ai.training.full_train --epochs 50 --mlm-epochs 5
```

GitHub Actions автоматически запускает отдельный full-training workflow на изменение его файла в экспериментальной ветке. Таймаут 350 минут, может быть превышен; отчет и лучший checkpoint сохраняются после каждой эпохи, upload при ошибке возможен, но hard timeout может прервать upload. `--resume` восстанавливает модель, optimizer, scheduler, scaler, epoch/step и RNG из `encoder-last.pt`, проверяя tokenizer и dataset SHA-256. Итоговая модель не публикуется в main автоматически.

## CUDA и локальный запуск

Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r ai/training/requirements-full.txt
.venv/bin/python -m ai.indexing.build_index
.venv/bin/python -m ai.training.prepare_full
ai/training/train_full.sh --epochs 50 --mlm-epochs 5 --device cuda --precision bf16
```

Windows: подготовить `.venv` через `py -3.12 -m venv .venv`, установить подходящий CUDA wheel PyTorch и `ai/training/requirements-full.txt`, затем вызвать `ai\training\train_full.bat --epochs 50 --mlm-epochs 5 --device cuda --precision bf16`.

Training по умолчанию требует CUDA; недоступный GPU приводит к ошибке, а не к тихому CPU fallback. `attention.py` использует PyTorch SDPA, который сам выбирает Flash, memory-efficient или math backend. `--precision bf16` проверяет поддержку BF16; `fp16` включает GradScaler. `--grad-accum N` включает накопление градиента. `--compile` оставлен opt-in: синтетический fixed-shape microbenchmark может быть быстрее, но этого недостаточно, чтобы объявить ускорение на динамически дополненных реальных батчах.

## Артефакты и фактический эксперимент

Не коммитить `ai/generated/`: там сохраняются `tokenizer.json`, `dataset/dataset-manifest.json`, `trained/encoder-mlm.pt`, `trained/encoder-best.pt`, `trained/encoder-last.pt`, `trained/training-report.json`, `trained/evaluation-report.json` и `trained/benchmark-report.json`.

Локальный эксперимент на RTX 4060: 5 MLM эпох, затем 50 контрастивных эпох, 9 100 шагов, 46 890 вариантов запросов; BF16/SDPA, batch 16, время 332.7 с. Best checkpoint — epoch 14. На Test encoder v0.2 получил Top-1 15.6%, Top-5 40.0%, MRR@5 24.6%; BM25F получил 77.0%, 98.6%, 86.1%. Validation выбрала lexical RRF weight 1.0, поэтому hybrid совпал с BM25F. Цели качества не достигнуты. E5 Small получил Top-1 64.1%, Top-5 91.2%, MRR@5 74.8% на тех же Test query/passage rows. Для v0.1 checkpoint в локальном workspace не найден, и метрики для него не выдумывались. Подробные цифры, latency, модельный размер и challenge queries — в `evaluation-report.json`; CUDA throughput и VRAM — в `benchmark-report.json`.

Отказ откалиброван только по Validation, добавив отдельные явно помеченные no-answer запросы в `ai/evaluation/unanswerable_validation.json`. Это небольшой искусственный набор; он не заменяет независимую проверку. Результаты typo/paraphrase/no-answer сохранены в challenge-секции evaluation report.

## Локальный браузерный просмотр

ONNX export проверяется против PyTorch; статический passage index строится из выбранного checkpoint. Файлы весов и индекса остаются в `ai/generated/` и не попадают в Git.

```bash
python -m ai.training.export_onnx
python -m ai.training.build_browser_assets --device cuda
npm --prefix web install
npm --prefix web run serve:ai
```

Откройте `http://localhost:8000/web/ai-search.html`. Страница выполняет запросы через ONNX Runtime Web/WASM; модель, tokenizer, индекс и runtime загружаются локально, без внешнего сервера или учителя. Для чистого checkout сначала надо получить локальные артефакты обучения; крупные веса намеренно не включены в Git.
