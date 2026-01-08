import cv2
import numpy as np
from ultralytics import YOLO


model_path = 'FINAL_EYELID_MODEL.pt'
try:
    model = YOLO(model_path)
except Exception as e:
    model = None
    print(f"Model yüklenemedi: {e}")


BLOOD_RED = (0, 0, 139)
VH_GREEN = (0, 128, 0)
POINT_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def px_per_cm_for_image(w, h):
    max_dim = max(w, h)
    if max_dim <= 640: return 16.0
    elif max_dim < 1600: return 80.0
    else: return 160.0

def px_to_cm(px, px_per_cm):
    return px / px_per_cm

def process_eye_image(image_bytes):
    if model is None:
        raise Exception("Model dosyası bulunamadı!")

    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Görüntü okunamadı.")

    h_img, w_img = img.shape[:2]
    max_dim = max(w_img, h_img)
    px_per_cm = px_per_cm_for_image(w_img, h_img)

    
    gscale = max(1.2, np.sqrt((w_img * h_img) / (1280 * 720)))
    line_thickness = max(3, int(4 * gscale))
    point_radius = max(4, int(6 * gscale))
    panel_scale_factor = 0.75 if 1000 <= max_dim <= 2000 else 1.0

    
    results = model.predict(source=img, conf=0.5, save=False)

    top_panel_lines = []
    bottom_panel_lines = []
    eye_centers = []

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        sorted_boxes = sorted(boxes, key=lambda b: float(b.xyxy[0][0] + b.xyxy[0][2]))
        labels = ["Left Eye", "Right Eye"]

        for idx, box in enumerate(sorted_boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, min(x1, w_img - 1))
            x2 = max(0, min(x2, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            y2 = max(0, min(y2, h_img - 1))
            conf = float(box.conf[0])

            cx, cy = np.round([(x1 + x2) / 2, (y1 + y2) / 2]).astype(int)
            eye_centers.append(cy)

            
            cv2.rectangle(img, (x1, y1), (x2, y2), BLOOD_RED, line_thickness)
            cv2.circle(img, (cx, cy), point_radius, POINT_COLOR, -1)

            mrd2_px = max(0.0, cy - y1)
            pfh_px = max(0.0, y2 - y1)
            mrd2_cm = px_to_cm(mrd2_px, px_per_cm)
            pfh_cm = px_to_cm(pfh_px, px_per_cm) * 0.75

            
            base_scale = min(x2 - x1, y2 - y1) / 180.0
            pct_scale = max(0.9, min(2.0, base_scale * 1.2 * gscale))
            pct_thick = max(2, int(round(2.0 * pct_scale)))
            pct_pad = max(4, int(round(5 * pct_scale)))
            pct_text = f"{conf*100:.1f}%"
            
            (w_pct, h_pct), b_pct = cv2.getTextSize(pct_text, FONT, pct_scale, pct_thick)
            badge_w = w_pct + 2 * pct_pad
            badge_h = h_pct + b_pct + 2 * pct_pad

            bx1 = x1 + pct_pad
            by1 = y2 + pct_pad
            if by1 + badge_h > h_img:
                by1 = y2 - badge_h - pct_pad

            overlay = img.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx1 + badge_w, by1 + badge_h), BLOOD_RED, -1)
            img[by1:by1+badge_h, bx1:bx1+badge_w] = cv2.addWeighted(
                overlay[by1:by1+badge_h, bx1:bx1+badge_w], 0.75,
                img[by1:by1+badge_h, bx1:bx1+badge_w], 0.25, 0
            )
            cv2.putText(img, pct_text, (bx1 + pct_pad, by1 + badge_h - pct_pad - b_pct),
                        FONT, pct_scale, (255, 255, 255), pct_thick, cv2.LINE_AA)

            name = labels[idx] if idx < len(labels) else f"Eye {idx+1}"
            top_panel_lines.append(f"{name}: MRD2 {mrd2_cm:.2f} cm | PFH {pfh_cm:.2f} cm")

    
    if len(eye_centers) >= 2:
        vh_px = abs(eye_centers[0] - eye_centers[1])
        vh_cm = px_to_cm(vh_px, px_per_cm)
        bottom_panel_lines.append(f"VH: {vh_cm:.2f} cm")

    
    if top_panel_lines:
        font_scale = max(1.0, min(2.0, (w_img + h_img) / 1800.0 * gscale)) * panel_scale_factor
        thick = max(2, int(round(2.5 * font_scale)))
        pad = max(6, int(round(8 * font_scale)))
        sizes = [cv2.getTextSize(t, FONT, font_scale, thick) for t in top_panel_lines]
        max_w = max(s[0][0] for s in sizes)
        line_h = max(s[0][1] + s[1] for s in sizes) + pad // 2
        panel_w = max_w + 2 * pad
        panel_h = len(top_panel_lines) * line_h + pad
        px1 = pad
        py1 = pad
        py2 = py1 + panel_h
        overlay = img.copy()
        cv2.rectangle(overlay, (px1, py1), (px1 + panel_w, py2), BLOOD_RED, -1)
        img[py1:py2, px1:px1+panel_w] = cv2.addWeighted(overlay[py1:py2, px1:px1+panel_w], 0.65, img[py1:py2, px1:px1+panel_w], 0.35, 0)
        for i, t in enumerate(top_panel_lines):
            tx = px1 + pad
            ty = py1 + pad + (i + 1) * line_h - sizes[i][1] - pad // 2
            cv2.putText(img, t, (tx, ty), FONT, font_scale, (255, 255, 255), thick, cv2.LINE_AA)

    
    if bottom_panel_lines:
        font_scale = max(1.0, min(2.0, (w_img + h_img) / 1800.0 * gscale)) * panel_scale_factor
        thick = max(2, int(round(2.5 * font_scale)))
        pad = max(6, int(round(8 * font_scale)))
        sizes = [cv2.getTextSize(t, FONT, font_scale, thick) for t in bottom_panel_lines]
        max_w = max(s[0][0] for s in sizes)
        line_h = max(s[0][1] + s[1] for s in sizes) + pad // 2
        panel_w = max_w + 2 * pad
        panel_h = len(bottom_panel_lines) * line_h + pad
        px1 = pad
        py1 = h_img - panel_h - pad
        if py1 < 0: py1 = pad
        py2 = py1 + panel_h
        overlay = img.copy()
        cv2.rectangle(overlay, (px1, py1), (px1 + panel_w, py2), VH_GREEN, -1)
        img[py1:py2, px1:px1+panel_w] = cv2.addWeighted(overlay[py1:py2, px1:px1+panel_w], 0.85, img[py1:py2, px1:px1+panel_w], 0.15, 0)
        for i, t in enumerate(bottom_panel_lines):
            tx = px1 + pad
            ty = py1 + pad + (i + 1) * line_h - sizes[i][1] - pad // 2
            cv2.putText(img, t, (tx, ty), FONT, font_scale, (255, 255, 255), thick, cv2.LINE_AA)

    
    _, encoded_img = cv2.imencode('.jpg', img)
    return encoded_img.tobytes()