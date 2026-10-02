# filename: vision.py
import torch
from torchvision import models, transforms
from PIL import Image
import config
import time
import cv2
import os
import numpy as np
import easyocr


class EnemyDetector:
    """Handles loading the YOLOv9 model and detecting enemy units."""
    def __init__(self, model_path, device):
        self.device = device
        self.model = self._load_model(model_path)
        self.class_names = self.model.names
        self.training_resolution = (640, 640)

    def _load_model(self, path):
        """Loads the custom-trained YOLOv9 model."""
        print("Loading enemy detector model...")
        try:
            # Using torch.hub.load to get the model
            model = torch.hub.load('WongKinYiu/yolov9', 'custom', path=path, trust_repo=True)
            model.to(self.device)
            model.eval() # Set the model to evaluation mode
            print("Enemy detector model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading enemy detector model from path '{path}': {e}")
            raise

    def detect_units(self, screenshot_pil):
        """Takes a PIL image, runs detection, and returns a list of detected units."""
        # Convert PIL image to numpy array for processing
        frame_rgb = np.array(screenshot_pil)
        original_height, original_width, _ = frame_rgb.shape
        
        # Preprocess: Resize the frame to the resolution the model was trained on
        resized_frame_rgb = cv2.resize(frame_rgb, self.training_resolution)
        
        # Run inference
        with torch.no_grad():
            results = self.model(resized_frame_rgb)
        
        detections = results.xyxy[0]

        # Calculate scaling factors to map detections back to the original image size
        x_scale = original_width / self.training_resolution[0]
        y_scale = original_height / self.training_resolution[1]

        detected_units = []
        for *box, conf, cls in detections:
            # Apply a confidence threshold to filter weak detections
            if conf >= 0.3: 
                # Scale box coordinates back to original frame size
                x1 = int(box[0] * x_scale)
                y1 = int(box[1] * y_scale)
                x2 = int(box[2] * x_scale)
                y2 = int(box[3] * y_scale)
                
                class_id = int(cls)
                class_name = self.class_names[class_id]
                
                # Append the detected unit's info to a list
                detected_units.append({
                    'name': class_name,
                    'confidence': float(conf),
                    'box': (x1, y1, x2, y2) # Bounding box in (left, top, right, bottom) format
                })
        return detected_units


# --- Component 1: The Logical Elixir Tracker ---
class ElixirTracker:
    """Tracks elixir logically, deducts card costs immediately, and syncs with visual reads."""
    def __init__(self):
        self.current_elixir = 5.0
        self.tracking_started = False
        self.last_update_time = None
        self.match_start_time = None

    def start(self):
        if not self.tracking_started:
            self.tracking_started = True
            self.current_elixir = 5.0
            self.last_update_time = time.time()
            self.match_start_time = time.time()
            print("💧 [ELIXIR] Hand detected! Initialized elixir tracking at 5.0")

    def reset(self):
        if self.tracking_started:
            self.tracking_started = False
            self.current_elixir = 0.0
            self.last_update_time = None
            self.match_start_time = None
            print("💧 [ELIXIR] Tracking paused.")

    def deduct(self, cost):
        """Immediately deducts elixir cost when a card is played."""
        if self.tracking_started and cost > 0:
            old = self.current_elixir
            self.current_elixir = max(0.0, self.current_elixir - cost)
            print(f"💧 [ELIXIR] Deducted {cost} for card. Elixir: {old:.1f} -> {self.current_elixir:.1f}")

    def update(self, elapsed_time):
        if not self.tracking_started: return
        match_elapsed = (time.time() - self.match_start_time) if self.match_start_time else 0
        rate = (1.0 / 1.4) if match_elapsed >= 120 else (1.0 / 2.8)
        self.current_elixir += elapsed_time * rate
        if self.current_elixir > 10.0:
            self.current_elixir = 10.0
        
    def sync_with_vision(self, visual_elixir):
        """Corrects the logical count when visual reading is confident."""
        if self.tracking_started and visual_elixir is not None and isinstance(visual_elixir, (int, float)):
            if 0 <= visual_elixir <= 10:
                diff = abs(self.current_elixir - float(visual_elixir))
                if diff >= 1.0:
                    print(f"💧 [ELIXIR SYNC] Correcting from {self.current_elixir:.1f} to {float(visual_elixir):.1f}")
                    self.current_elixir = float(visual_elixir)

# --- Component 2: The AI Card Classifier ---
class CardClassifier:
    """Handles loading the ML model and making predictions."""
    def __init__(self, model_path, class_names_path, device):
        self.device = device
        with open(class_names_path, "r") as f: self.class_names = [line.strip() for line in f.readlines()]
        self.transform = self._get_transform()
        self.model = self._load_model(model_path, len(self.class_names))

    def _load_model(self, path, num_classes):
        model = models.mobilenet_v2()
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        # Updated to use weights_only=True for safe loading, assuming it's a state dict
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        print("Card Classifier model loaded successfully.")
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize(config.IMG_SIZE), transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_batch(self, images):
        tensors = [self.transform(img) for img in images]
        if not tensors: return []
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)
            _, preds = torch.max(outputs, 1)
        return [self.class_names[p] for p in preds]
    
# --- ElixirVision Class (Integrated) ---
class ElixirVision:
    """
    Handles elixir detection using scaled template matching and color measurement.
    """
    def __init__(self, scaler=None):
        self.scaler = scaler
        self.template_dir = r"sorted_data/elixir"
        self.templates = self._load_elixir_templates()

    def _load_elixir_templates(self):
        templates = {}
        sx = getattr(self.scaler, 'x_scale', 1.0) if self.scaler else 1.0
        sy = getattr(self.scaler, 'y_scale', 1.0) if self.scaler else 1.0

        for i in range(11):
            path = os.path.join(self.template_dir, f"{i}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None and (abs(sx - 1.0) > 0.02 or abs(sy - 1.0) > 0.02):
                    nw = max(8, int(img.shape[1] * sx))
                    nh = max(8, int(img.shape[0] * sy))
                    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                templates[str(i)] = img
            else:
                print(f"Warning: Elixir template not found at {path}")
        return templates

    def get_elixir_value(self, screenshot_region):
        screenshot_gray = cv2.cvtColor(np.array(screenshot_region), cv2.COLOR_RGB2GRAY)
        
        best_match = None
        best_val = 0.55  # Adaptive threshold suitable for scaled templates
        
        for value, template in self.templates.items():
            if template is None:
                continue
            th, tw = template.shape
            sh, sw = screenshot_gray.shape
            if th > sh or tw > sw:
                continue

            res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                best_match = value
                
        return int(best_match) if best_match is not None else None

# --- Main Vision Class ---
class Vision:
    """A single class to handle all vision-related tasks."""
    def __init__(self, scaler):
        print("Initializing Vision...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        self.classifier = CardClassifier(config.MODEL_PATH, config.CLASS_NAMES_PATH, self.device)
        self.elixir_tracker = ElixirTracker()
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.elixir_vision = ElixirVision(scaler=scaler)
        
        # --- ADDED: Initialize the new enemy detector ---
        self.enemy_detector = EnemyDetector('enemy_boundary_detector.pt', self.device)
        
        self.last_visual_check_time = time.time()
        self.last_known_hand = []
        self.last_update_time = time.time()
    
    def _draw_debug_overlay(self, screenshot_cv, card_coordinates):
        """Draws all OCR and card boxes on a screenshot for debugging."""
        # Draw OCR regions (RED)
        for box in config.OCR_OFFSETS:
            x, y, w, h = box['x_offset'], box['y_offset'], box['width'], box['height']
            cv2.rectangle(screenshot_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(screenshot_cv, box['name'], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw card slots (BLUE)
        for i, box in enumerate(card_coordinates):
            x, y, w, h = box
            cv2.rectangle(screenshot_cv, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(screenshot_cv, f"CARD {i+1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        window_name = "Live Debugger"
        cv2.imshow(window_name, screenshot_cv)
        cv2.waitKey(1)

    def _get_ocr_values(self, screenshot):
        """Performs OCR on all predefined regions with resolution scaling and padding."""
        ocr_results = {}
        img_w, img_h = screenshot.size
        for box in config.OCR_OFFSETS:
            x = max(0, int(box['x_offset'] * self.scaler.x_scale) - 8)
            y = max(0, int(box['y_offset'] * self.scaler.y_scale) - 5)
            w = int(box['width'] * self.scaler.x_scale) + 20
            h = int(box['height'] * self.scaler.y_scale) + 25
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            region_image = screenshot.crop((x, y, x2, y2))
            img_np = np.array(region_image)
            result = self.reader.readtext(img_np, allowlist='0123456789:')
            text = ' '.join([res[1] for res in result]).strip()
            ocr_results[box['name']] = text
        return ocr_results

    def get_game_state(self, screenshot, card_coordinates):
        """
        The main method to get the current hand, elixir, OCR data, and now enemies.
        """
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time
        self.elixir_tracker.update(elapsed_time)
        self.last_update_time = current_time

        ocr_data = self._get_ocr_values(screenshot)
        
        if self.elixir_tracker.tracking_started and current_time - self.last_visual_check_time > 1.5:
            # Scaled elixir crop coordinates
            sx = getattr(self.scaler, 'x_scale', 1.0)
            sy = getattr(self.scaler, 'y_scale', 1.0)
            elixir_x = max(0, int(config.ELIXIR_OFFSET['x'] * sx) - 10)
            elixir_y = max(0, int(config.ELIXIR_OFFSET['y'] * sy) - 10)
            elixir_w = int(config.ELIXIR_OFFSET['width'] * sx) + 20
            elixir_h = int(config.ELIXIR_OFFSET['height'] * sy) + 20
            img_w, img_h = screenshot.size
            x2 = min(img_w, elixir_x + elixir_w)
            y2 = min(img_h, elixir_y + elixir_h)

            elixir_region = screenshot.crop((elixir_x, elixir_y, x2, y2))
            visual_elixir = self.elixir_vision.get_elixir_value(elixir_region)
            if visual_elixir is not None:
                self.elixir_tracker.sync_with_vision(visual_elixir)
            self.last_visual_check_time = current_time

        cards_to_predict = []
        for (x, y, w, h) in card_coordinates:
            cards_to_predict.append(screenshot.crop((x, y, x + w, y + h)))

        current_hand = self.classifier.predict_batch(cards_to_predict)
        
        invalid_slots = current_hand.count('empty') + current_hand.count('Unknown')
        if invalid_slots >= 2: 
            self.elixir_tracker.reset()
        elif invalid_slots == 0 and not self.elixir_tracker.tracking_started: 
            self.elixir_tracker.start()

        if current_hand != self.last_known_hand:
            self.last_known_hand = current_hand
        
        detected_enemies = self.enemy_detector.detect_units(screenshot)
            
        return {
            "hand": self.last_known_hand,
            "elixir": self.elixir_tracker.current_elixir,
            "ocr_data": ocr_data,
            "enemies": detected_enemies  # Add the list of enemies to the game state
        }
    
