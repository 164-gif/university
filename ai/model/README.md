# University Encoder v0.1 — этап 05

Реализация архитектуры, **без обученных семантических весов**. Собственные блоки внимания и Transformer на примитивах PyTorch. Готовая предобученная модель не используется.

| Параметр | Значение |
|---|---:|
| Словарь | Из tokenizer.json (16 384 на этапе 04) |
| Контекст | 512 |
| Hidden / embedding | 128 / 128 |
| Слои / головы | 4 / 4 |
| Размер головы | 32 |
| FFN | 512 |
| Dropout | 0.1 при обучении |
| Параметры при словаре 16 384 | 2 972 800 |

Поток: token embedding + обучаемая позиция → LayerNorm → 4 pre-norm блока двунаправленного self-attention и GELU FFN с residual → LayerNorm → masked mean pooling (включая BOS/EOS) → линейная проекция → L2-нормализация FP32. Правое заполнение игнорируется при внимании и pooling. Вектор можно сравнивать скалярным произведением после обучения. Все замаскированные строки дают нулевой вектор; нормальная текстовая последовательность содержит BOS/EOS.

```bash
python3 -m pip install -r ai/model/requirements.txt --index-url https://download.pytorch.org/whl/cpu
python3 -m ai.indexing.build_index
python3 -m ai.tokenizer.train
python3 -m unittest discover -s ai/tests -v
python3 -m ai.model.smoke
```

```python
from pathlib import Path
from ai.tokenizer.bpe import BPETokenizer
from ai.model.config import EncoderConfig
from ai.model.encoder import UniversityEncoder
from ai.model.inputs import collate_texts

tok = BPETokenizer.load(Path("ai/generated/tokenizer.json"))
config = EncoderConfig(vocab_size=tok.vocab_size)
model = UniversityEncoder(config)
batch = collate_texts(tok, ["Что такое скалярное произведение?"], config)
embedding = model(**batch)  # [1, 128], случайные веса до обучения
```

collate_texts динамически дополняет батч и отказывается от усечения. Длинные фрагменты обрабатываются на этапе датасета. checkpoint.py проверяет формат, конфигурацию и SHA-256 точного tokenizer.json; по умолчанию не загружает checkpoint с trained_steps=0. Это технический флаг, а не доказательство качества обучения. Не загружать недоверенные checkpoints.

smoke.py проверяет реальные входы, форму, конечность, норму, инвариантность padding и измеряет CPU forward с одним потоком после прогрева. Отчёт помечен random_untrained; время не включает загрузку, токенизацию или поиск. Теоретический объём INT8 не является измеренным размером ONNX. Экспорт/браузер и квантование ещё не реализованы.

Следующий этап — обучающая выборка, защита от утечки между train/validation/test, разделение длинных фрагментов. Для семантического поиска потребуется контрастивное обучение и сравнение с BM25F. Само наличие Transformer не обеспечивает понимание языка.

## RTX 4060 / v0.2

Текущий локальный checkpoint v0.2 имеет 2.84 млн параметров и использует tokenizer из `ai/generated/full/tokenizer.json`. Полный train/eval и воспроизводимые benchmark-команды описаны в [FULL_TRAINING.md](../training/FULL_TRAINING.md). На Test v0.2 пока существенно слабее BM25F; не использовать его как самостоятельный поисковый индекс для браузера до улучшения качества. Вес учителя E5 в компактную модель не входит.
