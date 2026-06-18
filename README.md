# Vzglyad-Detect

Русский | [English](#vzglyad-english)

Vzglyad - это независимая и легковесная библиотека компьютерного зрения для запуска моделей детекции объектов. Она позволяет использовать модели, обученные для Yolo, без необходимости использовать их тяжелые лицензии.

### Основные возможности

- **Ноль зависимостей от сторонних фреймворков на этапе чистого инференса.**
- **Автоматическое извлечение в оперативной памяти.** Можно загружать yolo модели.
- Легкая и быстрая архитектура.
- Простота интеграции в ваши существующие пайплайны компьютерного зрения.

### Использование

Вы можете загрузить оригинальные модели напрямую: библиотека автоматически извлечет веса в память. Вы также можете сразу сохранить очищенный `.pth` файл.

```python
import cv2
from vzglyad_vision import VisionPredictor

# Способ 1: Прямая загрузка оригинальной модели. 
# (Требует установленный ultralytics для первого запуска для данной модели)
# Вы можете сразу сохранить очищенные веса:
predictor = VisionPredictor("vasha_model.pt", device='cpu', save_extracted_path="clean_model.pth")

# Способ 2: Загрузка очищенных весов.
# predictor = VisionPredictor("clean_model.pth", device='cpu')

# Пример распознавания:
img = cv2.imread("test_image.jpg")
results, im0 = predictor.predict(img, conf_thres=0.25)

# Отрисовка результатов
img_drawn = predictor.draw_results(img, results)
cv2.imshow("Result", img_drawn)
cv2.waitKey(0)
```

### Лицензия (MIT)
Данная библиотека и использование "чистых" весов в формате `.pth` распространяются под лицензией **MIT**. Извлекая архитектуру и веса из сторонних фреймворков, вы можете использовать полученный `.pth` файл в коммерческих проектах без строгих копилефт-ограничений (таких как AGPL-3.0), если это не противоречит условиям лицензии исходных данных обучения. ! Использование .pt при каждом запуске в комерческих проектах требует ultralytics commerce лицензии. 

---

<h2 id="vzglyad-english">Vzglyad (English)</h2>

Vzglyad is an independent, lightweight vision library for object detection inference. It allows you to run detection models exported from other frameworks (like original `.pt` files) without keeping any heavy dependencies.

### Key Features

- **Zero dependencies on third-party frameworks during pure inference.**
- **Automatic in-memory extraction.** 
- Lightweight and fast architecture.
- Easy to integrate into your existing computer vision pipelines.

### Usage

You can load your original models directly, extract weights in memory, and optionally save a clean `.pth` file to completely drop the framework dependencies for future runs.

```python
import cv2
from vzglyad_vision import VisionPredictor

# Option 1: Load original model directly. 
# (Requires the ultralytics has to be installed just for this first run)
# We can save the clean weights simultaneously:
predictor = VisionPredictor("vasha_model.pt", device='cpu', save_extracted_path="clean_model.pth")

# Option 2: Load the extracted clean weights directly. 
# predictor = VisionPredictor("clean_model.pth", device='cpu')

# Run inference
img = cv2.imread("test_image.jpg")
results, im0 = predictor.predict(img, conf_thres=0.25)

# Draw results
img_drawn = predictor.draw_results(img, results)
cv2.imshow("Result", img_drawn)
cv2.waitKey(0)
```

### License (MIT)
This library and the use of "clean" weights in the `.pth` format are licensed under the **MIT** license. By extracting the architecture and weights from third-party frameworks, you can use the resulting `.pth` file in commercial projects without strict copyleft restrictions (such as AGPL-3.0), as long as it complies with the license terms of the original training data. Using the .pt file every time you run it in commercial projects requires the Ultralytics Commerce license.
