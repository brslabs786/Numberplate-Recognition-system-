import os
import re
import cv2
import numpy as np
import easyocr

# Set model directory within project
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'easyocr')

# Initialize the OCR reader with English and local model directory
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=MODEL_DIR, user_network_directory=MODEL_DIR)

# Valid Indian State and UT codes
INDIAN_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN',
    'GA', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD',
    'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ',
    'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'UT', 'WB'
}

# Common OCR confusion for State code prefix
STATE_FUZZY = {
    'M8': 'MH', '0D': 'OD', 'D1': 'DL', '7N': 'TN', '1N': 'TN',
    'U8': 'UP', 'K4': 'KA', 'H8': 'HR', 'G1': 'GJ', 'R1': 'RJ',
    'W8': 'WB', 'P8': 'PB', 'B8': 'BR', '0L': 'DL', 'DI': 'DL',
    'O0': 'OD', '05': 'OS', '7S': 'TS', '1S': 'TS'
}


def to_digits(s: str) -> str:
    """
    Corrects letter-like characters to digits strictly in numeric slots (RTO, 4-digit number, year).
    NOTE: '2' stays '2', '1' stays '1'. 'T' is corrected to '7' (not 1).
    """
    char_to_digit_map = {
        'O': '0', 'D': '0', 'Q': '0',
        'I': '1', 'L': '1', 'J': '1',
        'Z': '2',
        'E': '3',
        'A': '4',
        'S': '5',
        'G': '6', 'b': '6',
        'T': '7',  # T looks like 7 in digital/embossed font
        'B': '8',
        'P': '9', 'q': '9'
    }
    return "".join(char_to_digit_map.get(c, c) for c in s)


def to_letters(s: str) -> str:
    """
    Corrects digit-like characters to letters strictly in alphabetic slots (State code, Series).
    NOTE: 'T' stays 'T', 'Z' stays 'Z'. Digits become letters only where letters belong.
    """
    digit_to_char_map = {
        '0': 'O',
        '1': 'I',
        '2': 'Z',
        '3': 'J',
        '4': 'A',
        '5': 'S',
        '6': 'G',
        '7': 'T',
        '8': 'B'
    }
    return "".join(digit_to_char_map.get(c, c) for c in s)


def clean_text(raw_text: str) -> str:
    """Removes non-alphanumeric characters and converts to uppercase."""
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
    # Strip leading IND tag often found on HSRP plates
    if cleaned.startswith('IND') and len(cleaned) > 10:
        cleaned = cleaned[3:]
    return cleaned


def is_valid_indian_plate(text: str) -> bool:
    """Validates if formatted text conforms to Indian registration."""
    cleaned = clean_text(text)
    if len(cleaned) < 7 or len(cleaned) > 12:
        return False

    # Bharat (BH) series: e.g. 22BH1234AA
    if re.match(r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$', cleaned):
        return True

    # Standard state registration regex
    standard_pattern = r'^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{4})$'
    match = re.match(standard_pattern, cleaned)
    if match:
        state = match.group(1)
        if state in INDIAN_STATE_CODES:
            return True
        return True

    return False


def format_indian_license(raw_text: str) -> str:
    """
    Context-aware disambiguation and formatting for Indian number plates:
    - Never changes 'T' to '1' in letter slots.
    - Never changes '2' to 'Z' in digit slots.
    - Accurately splits into [State, RTO, Series, Number].
    """
    cleaned = clean_text(raw_text)
    if len(cleaned) < 7:
        return cleaned

    # 1. Check Bharat (BH) Series: e.g. 22 BH 1234 AA
    if 'BH' in cleaned:
        idx = cleaned.find('BH')
        if idx >= 2 and len(cleaned) >= idx + 6:
            year = to_digits(cleaned[idx-2:idx])
            num = to_digits(cleaned[idx+2:idx+6])
            series = to_letters(cleaned[idx+6:]) if len(cleaned) > idx+6 else ''
            if series:
                return f"{year} BH {num} {series}".strip()
            else:
                return f"{year} BH {num}".strip()

    # 2. Standard format: State(2) + RTO(1-2) + Series(0-3) + Number(4)
    if len(cleaned) >= 8:
        # State Code (first 2 chars)
        state_raw = cleaned[:2]
        if state_raw in STATE_FUZZY:
            state = STATE_FUZZY[state_raw]
        else:
            state = to_letters(state_raw)
            if state in STATE_FUZZY:
                state = STATE_FUZZY[state]

        # 4-Digit Unique Number (last 4 chars)
        num_raw = cleaned[-4:]
        number = to_digits(num_raw)

        # Middle part (between state and 4-digit number)
        # Contains RTO code (1 or 2 digits) followed by Series (0 to 3 letters)
        mid_raw = cleaned[2:-4]
        rto_digits = []
        series_chars = []

        for i, c in enumerate(mid_raw):
            # First 1-2 characters in middle are typically RTO digits
            if i < 2 and (c.isdigit() or (len(rto_digits) < 2 and not series_chars and c in 'OIZSEASGBT')):
                if i == 0 or (i == 1 and not series_chars):
                    rto_digits.append(to_digits(c))
                    continue
            series_chars.append(to_letters(c))

        rto = "".join(rto_digits)
        series = "".join(series_chars)

        parts = [state]
        if rto:
            parts.append(rto)
        if series:
            parts.append(series)
        parts.append(number)
        return " ".join(parts)

    return cleaned


def preprocess_plate(plate_crop):
    """
    Produces multiple image variations for robust OCR:
    1. Grayscale
    2. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
    3. Bilateral Filtered
    4. Adaptive Gaussian Threshold
    5. Otsu Threshold
    """
    if plate_crop is None or plate_crop.size == 0:
        return []

    h, w = plate_crop.shape[:2]
    # Enlarge plate crop if it's small for clearer OCR character segmentation
    if h < 70 or w < 180:
        scale = max(2.5, 140.0 / max(1, h))
        plate_crop = cv2.resize(plate_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE Contrast Boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Bilateral Filter
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # 3. Adaptive Threshold
    adaptive = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)

    # 4. Otsu Threshold
    _, otsu = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [gray, enhanced, filtered, adaptive, otsu]


def extract_combined_text(detections):
    """
    Extracts and merges multi-box detections (such as 2-line number plates).
    Sorts top-to-bottom (Y), then left-to-right (X).
    """
    if not detections:
        return ""

    boxes = []
    for d in detections:
        bbox, text, score = d
        if score < 0.15:
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x_min = min(xs)
        y_center = (min(ys) + max(ys)) / 2
        boxes.append({
            'text': text,
            'x': x_min,
            'y': y_center,
            'score': score
        })

    if not boxes:
        return ""

    # Cluster into lines
    boxes.sort(key=lambda b: b['y'])
    lines = []
    current_line = [boxes[0]]
    line_y = boxes[0]['y']

    for b in boxes[1:]:
        if abs(b['y'] - line_y) < 22:
            current_line.append(b)
        else:
            lines.append(current_line)
            current_line = [b]
            line_y = b['y']
    lines.append(current_line)

    full_text = []
    for line in lines:
        line.sort(key=lambda b: b['x'])
        full_text.extend([b['text'] for b in line])

    return " ".join(full_text)


def read_license_plate(license_plate_crop):
    """
    Read license plate text from cropped image using EasyOCR with character allowlist.
    Tries multiple preprocessing techniques and returns the best formatted plate string.
    """
    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, None

    candidates = []
    crops_to_try = preprocess_plate(license_plate_crop)
    allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for img in crops_to_try:
        try:
            detections = reader.readtext(img, detail=1, paragraph=False, allowlist=allowlist)
            if not detections:
                continue

            # 1. Try merged multi-line text (for 2-line or spaced plates)
            combined_raw = extract_combined_text(detections)
            combined_cleaned = clean_text(combined_raw)
            if len(combined_cleaned) >= 7:
                is_valid = is_valid_indian_plate(combined_cleaned)
                formatted = format_indian_license(combined_cleaned)
                avg_score = sum(d[2] for d in detections) / len(detections)
                adjusted_score = avg_score * (1.5 if is_valid else 1.0)
                candidates.append((formatted, avg_score, is_valid, adjusted_score))

            # 2. Also test individual detections
            for d in detections:
                bbox, text, score = d
                cleaned = clean_text(text)
                if len(cleaned) >= 7:
                    is_valid = is_valid_indian_plate(cleaned)
                    formatted = format_indian_license(cleaned)
                    adjusted_score = score * (1.5 if is_valid else 1.0)
                    candidates.append((formatted, score, is_valid, adjusted_score))

        except Exception:
            continue

    if not candidates:
        # Fallback without allowlist
        try:
            detections = reader.readtext(license_plate_crop, detail=1)
            for d in detections:
                bbox, text, score = d
                cleaned = clean_text(text)
                if len(cleaned) >= 5:
                    is_valid = is_valid_indian_plate(cleaned)
                    formatted = format_indian_license(cleaned)
                    candidates.append((formatted, score, is_valid, score))
        except Exception:
            pass

    if not candidates:
        return None, None

    # Sort by: valid Indian format first, then adjusted score
    candidates.sort(key=lambda c: (1 if c[2] else 0, c[3]), reverse=True)
    best = candidates[0]
    return best[0], float(best[1])


def write_csv(results, output_path):
    """Write the results to a CSV file."""
    with open(output_path, 'w') as f:
        f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                item = results[frame_nmr][car_id]
                if 'car' in item and 'license_plate' in item and 'text' in item['license_plate']:
                    car_box = item['car']['bbox']
                    plate_box = item['license_plate']['bbox']
                    f.write('{},{},[{} {} {} {}],[{} {} {} {}],{},{},{}\n'.format(
                        frame_nmr,
                        car_id,
                        car_box[0], car_box[1], car_box[2], car_box[3],
                        plate_box[0], plate_box[1], plate_box[2], plate_box[3],
                        item['license_plate']['bbox_score'],
                        item['license_plate']['text'],
                        item['license_plate']['text_score']
                    ))


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        p_cx = (x1 + x2) / 2
        p_cy = (y1 + y2) / 2
        if xcar1 <= p_cx <= xcar2 and ycar1 <= p_cy <= ycar2:
            return vehicle_track_ids[j]

        if (x1 >= xcar1 - 10) and (y1 >= ycar1 - 10) and (x2 <= xcar2 + 10) and (y2 <= ycar2 + 10):
            return vehicle_track_ids[j]

    return -1, -1, -1, -1, -1
