# Vzglyad-Detect

Русский | [English](#vzglyad-english)

Vzglyad - это независимая и легковесная библиотека компьютерного зрения для запуска моделей детекции объектов. Она позволяет использовать модели, обученные для Yolo, без необходимости использовать их тяжелые лицензии.

### Основные возможности

- **Ноль зависимостей от сторонних фреймворков на этапе чистого инференса.**
- **Конвертация моделей.** Вы можете использовать отдельный инструмент [YoloPT-to-VZglyad-detectPTH](https://github.com/AlexeyRF/YoloPT-to-VZglyad-detectPTH) для перевода .pt в .pth !! Прочтите ридми там важно.
- Легкая и быстрая архитектура.
- Простота интеграции в ваши существующие пайплайны компьютерного зрения.

### Использование

Для использования библиотеки необходимо предварительно конвертировать вашу оригинальную модель (.pt) в формат (.pth) с помощью отдельного конвертера [YoloPT-to-VZglyad-detectPTH](https://github.com/AlexeyRF/YoloPT-to-VZglyad-detectPTH).

```python
import cv2
from vzglyad_vision import VisionPredictor

# Загрузка весов
predictor = VisionPredictor("clean_model.pth", device='cpu')

# Пример распознавания:
img = cv2.imread("test_image.jpg")
results, im0 = predictor.predict(img, conf_thres=0.25)

# Отрисовка результатов
img_drawn = predictor.draw_results(img, results)
cv2.imshow("Result", img_drawn)
cv2.waitKey(0)
```

### Лицензия (MIT)
Данная библиотека и использование "чистых" весов в формате `.pth` распространяются под лицензией **MIT**. 

---

<h2 id="vzglyad-english">Vzglyad (English)</h2>

Vzglyad is an independent, lightweight vision library for object detection inference. 

### Key Features

- **Zero dependencies on third-party frameworks during pure inference.**
- **Model Conversion.** You can use the separate [YoloPT-to-VZglyad-detectPTH](https://github.com/AlexeyRF/YoloPT-to-VZglyad-detectPTH) converter tool to translate .pt to .pth.
- Lightweight and fast architecture.
- Easy to integrate into your existing computer vision pipelines.

### Usage

To use the library, you must first convert your original model (.pt) to the (.pth) format using the separate converter tool [YoloPT-to-VZglyad-detectPTH](https://github.com/AlexeyRF/YoloPT-to-VZglyad-detectPTH).

```python
import cv2
from vzglyad_vision import VisionPredictor

# Load the extracted clean weights directly (doesn't violate AGPL-3). 
predictor = VisionPredictor("clean_model.pth", device='cpu')

# Run inference
img = cv2.imread("test_image.jpg")
results, im0 = predictor.predict(img, conf_thres=0.25)

# Draw results
img_drawn = predictor.draw_results(img, results)
cv2.imshow("Result", img_drawn)
cv2.waitKey(0)
```

### License (MIT)
This library and the use of "clean" weights in the `.pth` format are licensed under the **MIT** license. 