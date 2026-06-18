import cv2
import numpy as np
from mss import mss
from vzglyad_vision import VisionPredictor

def main():
    try:
        predictor = VisionPredictor(r"path\to\your\model", device='cpu')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    sct = mss()
    
    # Получаем информацию о главном мониторе
    primary_monitor = sct.monitors[1]
    
    width = 640
    height = 640
    
    # Вычисляем координаты для захвата области 640x640 (по умолчанию по центру экрана)
    # Вы можете изменить значения 'left' и 'top', чтобы захватывать другую область.
    left = primary_monitor["left"] + (primary_monitor["width"] - width) // 2
    top = primary_monitor["top"] + (primary_monitor["height"] - height) // 2
    
    monitor = {
        "top": top,
        "left": left,
        "width": width,
        "height": height
    }

    print(f"Захват области: 640x640 (Left: {left}, Top: {top})")

    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Передаем кадр в модель (размер кадра уже 640x640)
        results, im0 = predictor.predict(frame, conf_thres=0.05)
        annotated_frame = predictor.draw_results(frame, results)
        
        # Поскольку окно теперь 640x640, мы можем показывать его без уменьшения масштаба
        cv2.imshow('VZGLYAD Real-Time Screen Analysis (640x640)', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
