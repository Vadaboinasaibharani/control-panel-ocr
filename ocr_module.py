# ocr_module.py
import cv2
import numpy as np
import re
import tempfile
import pytesseract
import warnings
import os
warnings.filterwarnings("ignore")

# Set Tesseract path on Linux/Render
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def detect_display_bbox(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 20, 20])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        best, best_area = None, 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 400: continue
            ar = w / float(h)
            if 1.2 <= ar <= 12 and area > best_area:
                best = (x, y, w, h); best_area = area
        if best:
            return best
    H, W = img.shape[:2]
    w, h = int(W * 0.7), int(H * 0.25)
    x, y = (W - w)//2, (H - h)//2
    return (x, y, w, h)

def preprocess_roi(roi, upscale=2):
    if upscale != 1:
        roi = cv2.resize(roi, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)
    th = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 9)
    return enhanced, th

def ocr_tesseract_image(img_gray):
    data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT, config='--psm 6')
    out = []
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        if not txt: continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        try: conf = float(data['conf'][i])
        except: conf = -1.0
        bbox = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        out.append((bbox, txt, conf/100.0))
    return out

def clean_token(tok):
    tok = tok.strip().replace(',', '.')
    if re.search(r'\d', tok):
        tok = tok.replace('O','0').replace('o','0').replace('l','1').replace('I','1').replace('B','8')
    return tok

def reconstruct_from_bboxes(results):
    tokens = []
    for bbox, text, _ in results:
        try: cx = int(sum([p[0] for p in bbox])/len(bbox))
        except: cx = 0
        tokens.append((text, cx))
    tokens.sort(key=lambda t: t[1])
    return tokens

def parse_numeric(tokens, joined_raw):
    joined_raw = joined_raw.replace(',', '.')
    m = re.search(r'\d+\.\d+', joined_raw)
    if m: return float(m.group())
    nums = [clean_token(t) for t,_ in tokens if re.search(r'\d', t)]
    if not nums: return None
    joined = ''.join(nums)
    if '.' not in joined and len(joined) >= 4:
        return round(float(joined[:-2] + '.' + joined[-2:]), 4)
    try: return float(joined)
    except: return None

def correct_numeric(value, label):
    if value is None: return None
    if label == 'Pressure' and value > 20:
        for d in (1000,100,10):
            v = value/d
            if 0 < v < 20: return round(v,4)
    if label == 'Level' and value > 100:
        for d in (100,10):
            v = value/d
            if 0 < v <= 100: return round(v,4)
    return round(value,4)

def detect_label(joined, value):
    text = joined.lower()
    if any(k in text for k in ['he level','heleve','level','leve1','%']): return 'Level'
    if any(k in text for k in ['he pressure','pressure','psi']): return 'Pressure'
    if value is not None:
        if value <=100: return 'Level'
        else: return 'Pressure'
    return None

def analyze_image(path):
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(f"Could not load image: {path}")
    x,y,w,h = detect_display_bbox(img)
    roi = img[y:y+h, x:x+w]
    enhanced, th = preprocess_roi(roi)
    results = ocr_tesseract_image(th)
    joined = " | ".join([r[1] for r in results])
    tokens = reconstruct_from_bboxes(results)
    value = parse_numeric(tokens, joined)
    label = detect_label(joined, value)
    value = correct_numeric(value, label)

    # Annotate image
    annotated = img.copy()
    for bbox, text, _ in results:
        try: tl, br = (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1]))
        except: tl, br = (0,0),(0,0)
        cv2.rectangle(annotated, tl, br, (0,128,255),2)
        cv2.putText(annotated, str(text), (tl[0], max(10, tl[1]-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255),2)

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    annotated_path = tf.name
    tf.close()
    cv2.imwrite(annotated_path, annotated)

    return {'path': path, 'label': label, 'value': value, 'raw_text': joined, 'annotated_path': annotated_path}
