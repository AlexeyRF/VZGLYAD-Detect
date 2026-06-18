import cv2
from vzglyad_vision import VisionPredictor

def main():
    print("Запуск модуля...")
    
    predictor = VisionPredictor(r"path\to\your\model", device='cpu')
    
    img_path = r"path\to\your\image.png"
    import numpy as np
    import os
    if not os.path.exists(img_path):
        print(f"Ошибка: Файл не найден по пути: {img_path}")
        return
        
    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"Ошибка: Не удалось прочитать изображение: {img_path}")
        return
        
    results, im0 = predictor.predict(img, conf_thres=0.015)
    
    img_drawn = predictor.draw_results(img, results)
    cv2.imshow("Result", img_drawn)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
