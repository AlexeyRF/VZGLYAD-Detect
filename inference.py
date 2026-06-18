import torch
import cv2
import numpy as np
from .model import CustomVisionModel
from .utils import letterbox, non_max_suppression, scale_boxes

class VisionPredictor:
    """
    Main prediction interface for the independent vision module.
    """
    def __init__(self, model_path, device=None, imgsz=640):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.imgsz = imgsz
        
        # Load the file
        data = torch.load(model_path, map_location=self.device)
        
        # Check if it's an original ultralytics model
        if 'model' in data and 'yaml' not in data:
            raise ValueError("Loading original .pt models directly is not supported to avoid AGPL-3 dependencies. "
                             "Please use the standalone converter tool to convert your .pt model to .pth.")
        else:
            self.yaml = data.get('yaml')
            self.metadata = data.get('metadata')
            state_dict = data.get('state_dict')
            
        if self.yaml is None:
            raise ValueError("Could not extract YAML configuration from the model.")
        
        # Build the model
        self.model = CustomVisionModel(self.yaml).to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        # Set strides for the Detect head
        self.model.model[-1].stride = torch.tensor(self.metadata['stride']).to(self.device)
        self.names = self.metadata['names']
        
    def preprocess(self, img):
        """Prepares a BGR numpy image for inference."""
        im0 = img.copy()
        im, _, _ = letterbox(im0, self.imgsz, auto=False, stride=32)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im_tensor = torch.from_numpy(im).to(self.device).float()
        im_tensor /= 255.0
        if len(im_tensor.shape) == 3:
            im_tensor = im_tensor[None]
        return im_tensor, im0, im.shape[1:]

    def predict(self, img_bgr, conf_thres=0.25, iou_thres=0.45):
        """Runs inference and post-processing."""
        im_tensor, im0, _ = self.preprocess(img_bgr)
        
        with torch.no_grad():
            preds = self.model(im_tensor)
            
        # NMS
        results = non_max_suppression(preds, conf_thres, iou_thres)[0]
        
        # Scale back to original image
        if len(results):
            results[:, :4] = scale_boxes(im_tensor.shape[2:], results[:, :4], im0.shape).round()
            
        return results, im0
        
    def draw_results(self, img, results):
        """Helper to draw boxes and labels."""
        img_out = img.copy()
        for *xyxy, conf, cls in results:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{self.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img_out
