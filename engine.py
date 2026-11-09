import os
import cv2
import numpy as np
import base64
import time
from collections import defaultdict
from ultralytics import YOLO
import util
from sort.sort import Sort

CLASS_NAMES = {
    1: 'Bicycle',
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck'
}

STATE_NAMES = {
    'AN': 'Andaman & Nicobar', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh',
    'AS': 'Assam', 'BR': 'Bihar', 'CG': 'Chhattisgarh', 'CH': 'Chandigarh',
    'DD': 'Daman & Diu', 'DL': 'Delhi', 'DN': 'Dadra & Nagar Haveli',
    'GA': 'Goa', 'GJ': 'Gujarat', 'HP': 'Himachal Pradesh', 'HR': 'Haryana',
    'JH': 'Jharkhand', 'JK': 'Jammu & Kashmir', 'KA': 'Karnataka', 'KL': 'Kerala',
    'LA': 'Ladakh', 'LD': 'Lakshadweep', 'MH': 'Maharashtra', 'ML': 'Meghalaya',
    'MN': 'Manipur', 'MP': 'Madhya Pradesh', 'MZ': 'Mizoram', 'NL': 'Nagaland',
    'OD': 'Odisha', 'PB': 'Punjab', 'PY': 'Puducherry', 'RJ': 'Rajasthan',
    'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TR': 'Tripura', 'TS': 'Telangana',
    'UK': 'Uttarakhand', 'UP': 'Uttar Pradesh', 'UT': 'Uttarakhand', 'WB': 'West Bengal',
    'BH': 'Bharat Series (All-India)'
}


def get_state_from_plate(plate_text: str) -> str:
    cleaned = util.clean_text(plate_text)
    if 'BH' in cleaned[:4]:
        return STATE_NAMES['BH']
    prefix = cleaned[:2]
    return STATE_NAMES.get(prefix, 'India')


def cv2_to_base64(img, quality=85):
    """Encodes OpenCV image to base64 jpeg data URL."""
    if img is None or img.size == 0:
        return ""
    success, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"


def filter_overlapping_vehicles(vehicles, iou_thresh=0.45):
    """Removes duplicate vehicle boxes for the same physical vehicle."""
    if not vehicles:
        return []
    vehicles = sorted(vehicles, key=lambda v: v['score'], reverse=True)
    keep = []
    for v in vehicles:
        overlap = False
        vx1, vy1, vx2, vy2 = v['bbox']
        v_area = max(1, (vx2 - vx1) * (vy2 - vy1))
        for k in keep:
            kx1, ky1, kx2, ky2 = k['bbox']
            k_area = max(1, (kx2 - kx1) * (ky2 - ky1))
            ix1, iy1 = max(vx1, kx1), max(vy1, ky1)
            ix2, iy2 = min(vx2, kx2), min(vy2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                iou = inter / float(v_area + k_area - inter)
                if iou > iou_thresh or inter / min(v_area, k_area) > 0.70:
                    overlap = True
                    break
        if not overlap:
            keep.append(v)
    return keep


def draw_styled_box(img, x1, y1, x2, y2, color=(0, 230, 118), label=None, thickness=2, corner_length=20):
    """Draws sleek corner brackets and semi-transparent label badge."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w - 1, int(x2)), min(h - 1, int(y2))
    cl = min(corner_length, (x2 - x1) // 3, (y2 - y1) // 3)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    ct = thickness + 2
    cv2.line(img, (x1, y1), (x1 + cl, y1), color, ct)
    cv2.line(img, (x1, y1), (x1, y1 + cl), color, ct)
    cv2.line(img, (x2, y1), (x2 - cl, y1), color, ct)
    cv2.line(img, (x2, y1), (x2, y1 + cl), color, ct)
    cv2.line(img, (x1, y2), (x1 + cl, y2), color, ct)
    cv2.line(img, (x1, y2), (x1, y2 - cl), color, ct)
    cv2.line(img, (x2, y2), (x2 - cl, y2), color, ct)
    cv2.line(img, (x2, y2), (x2, y2 - cl), color, ct)

    if label:
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        bx1 = x1
        by1 = max(0, y1 - th - 10)
        bx2 = min(w - 1, x1 + tw + 14)
        by2 = y1

        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, -1)
        cv2.putText(img, label, (bx1 + 7, by2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)


class ANPREngine:
    def __init__(self, coco_path="yolov8n.pt", plate_path="models/license_plate_detector.pt"):
        print(f"Loading vehicle detection model from {coco_path}...")
        self.coco_model = YOLO(coco_path)
        print(f"Loading license plate model from {plate_path}...")
        self.plate_model = YOLO(plate_path)
        self.vehicle_classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

    def detect_image(self, frame):
        """
        Detects EVERY vehicle on screen ("by hook or by crook").
        Associates plates when present, or provides vehicle tracking details when plate is not visible.
        """
        start_time = time.time()
        annotated = frame.copy()
        height, width = frame.shape[:2]

        # 1. Detect ALL vehicles with high sensitivity (conf=0.12)
        vehicle_res = self.coco_model(frame, conf=0.12, verbose=False)[0]
        raw_vehicles = []
        for box in vehicle_res.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if int(class_id) in self.vehicle_classes:
                raw_vehicles.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score),
                    'class_name': CLASS_NAMES.get(int(class_id), 'Vehicle')
                })

        # Remove duplicate bounding boxes on the same physical vehicle
        vehicles = filter_overlapping_vehicles(raw_vehicles)

        # 2. Detect ALL license plates with high sensitivity (conf=0.15)
        plate_res = self.plate_model(frame, conf=0.15, verbose=False)[0]
        plate_boxes = []
        for p_box in plate_res.boxes.data.tolist():
            px1, py1, px2, py2, p_score, _ = p_box
            px1_c, py1_c = max(0, int(px1)), max(0, int(py1))
            px2_c, py2_c = min(width, int(px2)), min(height, int(py2))
            if (px2_c - px1_c) > 15 and (py2_c - py1_c) > 8:
                plate_boxes.append({
                    'bbox': [px1_c, py1_c, px2_c, py2_c],
                    'score': float(p_score)
                })

        # 3. Associate plates to vehicles
        # Each vehicle will get its plate if present, or None
        vehicle_plate_map = {}
        assigned_plates = set()

        for v_idx, v in enumerate(vehicles):
            vx1, vy1, vx2, vy2 = v['bbox']
            best_plate = None
            best_dist = float('inf')

            for p_idx, p in enumerate(plate_boxes):
                if p_idx in assigned_plates:
                    continue
                px1, py1, px2, py2 = p['bbox']
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2

                # Check if plate center is inside vehicle box
                if (vx1 - 15) <= pcx <= (vx2 + 15) and (vy1 - 15) <= pcy <= (vy2 + 15):
                    dist = abs(pcx - (vx1 + vx2)/2) + abs(pcy - (vy1 + vy2)/2)
                    if dist < best_dist:
                        best_dist = dist
                        best_plate = (p_idx, p)

            if best_plate:
                vehicle_plate_map[v_idx] = best_plate[1]
                assigned_plates.add(best_plate[0])

        # 4. Process EVERY vehicle and generate result card
        results_list = []

        for v_idx, v in enumerate(vehicles):
            v_id = v_idx + 1
            vx1, vy1, vx2, vy2 = [int(c) for c in v['bbox']]
            vx1_c, vy1_c = max(0, vx1), max(0, vy1)
            vx2_c, vy2_c = min(width, vx2), min(height, vy2)

            v_crop = frame[vy1_c:vy2_c, vx1_c:vx2_c] if (vx2_c > vx1_c and vy2_c > vy1_c) else None
            v_crop_b64 = cv2_to_base64(v_crop)

            has_plate = v_idx in vehicle_plate_map
            if has_plate:
                p = vehicle_plate_map[v_idx]
                px1, py1, px2, py2 = p['bbox']
                plate_crop = frame[py1:py2, px1:px2]
                plate_crop_b64 = cv2_to_base64(plate_crop)

                # Attempt OCR with multi-level preprocessing
                text, ocr_score = util.read_license_plate(plate_crop)
                if not text or len(util.clean_text(text)) < 4:
                    text = "PLATE DETECTED"
                    ocr_score = 0.0

                state_name = get_state_from_plate(text) if text != "PLATE DETECTED" else "India"

                # Draw vehicle box (Blue)
                draw_styled_box(annotated, vx1, vy1, vx2, vy2, color=(59, 130, 246), label=f"{v['class_name']} #{v_id}", thickness=2)

                # Draw plate box (Amber)
                draw_styled_box(annotated, px1, py1, px2, py2, color=(0, 215, 255), label="Plate", thickness=2, corner_length=12)

                # Draw HSRP Banner
                banner_text = f"IND | {text}"
                (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                banner_y = max(th + 14, int(py1) - 10)
                banner_x1 = max(0, int(px1))
                banner_x2 = min(width - 1, banner_x1 + tw + 20)

                cv2.rectangle(annotated, (banner_x1, banner_y - th - 12), (banner_x2, banner_y), (255, 255, 255), -1)
                cv2.rectangle(annotated, (banner_x1, banner_y - th - 12), (banner_x2, banner_y), (0, 0, 0), 2)
                cv2.rectangle(annotated, (banner_x1, banner_y - th - 12), (banner_x1 + 45, banner_y), (180, 50, 0), -1)
                cv2.putText(annotated, "IND", (banner_x1 + 6, banner_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(annotated, text, (banner_x1 + 52, banner_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)

                results_list.append({
                    'vehicle_id': v_id,
                    'vehicle_type': v['class_name'],
                    'vehicle_score': round(v['score'], 2),
                    'vehicle_crop_base64': v_crop_b64,
                    'has_plate': True,
                    'plate_number': text,
                    'state': state_name,
                    'ocr_score': round(float(ocr_score), 2) if ocr_score else 0.0,
                    'plate_score': round(float(p['score']), 2),
                    'plate_crop_base64': plate_crop_b64,
                    'plate_bbox': [px1, py1, px2, py2],
                    'vehicle_bbox': [vx1, vy1, vx2, vy2]
                })
            else:
                # Vehicle detected but plate is obscured / side angle / out of view
                # WE STILL INCLUDE THIS VEHICLE SO NO VEHICLE IS DROPPED!
                draw_styled_box(annotated, vx1, vy1, vx2, vy2, color=(168, 85, 247), label=f"{v['class_name']} #{v_id}", thickness=2)

                results_list.append({
                    'vehicle_id': v_id,
                    'vehicle_type': v['class_name'],
                    'vehicle_score': round(v['score'], 2),
                    'vehicle_crop_base64': v_crop_b64,
                    'has_plate': False,
                    'plate_number': "Plate Not in View",
                    'state': "-",
                    'ocr_score': 0.0,
                    'plate_score': 0.0,
                    'plate_crop_base64': "",
                    'plate_bbox': None,
                    'vehicle_bbox': [vx1, vy1, vx2, vy2]
                })

        # Also add any standalone plate that wasn't attached to a vehicle box
        for p_idx, p in enumerate(plate_boxes):
            if p_idx not in assigned_plates:
                px1, py1, px2, py2 = p['bbox']
                plate_crop = frame[py1:py2, px1:px2]
                text, ocr_score = util.read_license_plate(plate_crop)
                if not text:
                    text = "PLATE DETECTED"
                draw_styled_box(annotated, px1, py1, px2, py2, color=(0, 215, 255), label="Plate", thickness=2)
                results_list.append({
                    'vehicle_id': len(results_list) + 1,
                    'vehicle_type': 'Vehicle',
                    'vehicle_score': 0.70,
                    'vehicle_crop_base64': "",
                    'has_plate': True,
                    'plate_number': text,
                    'state': get_state_from_plate(text),
                    'ocr_score': round(float(ocr_score or 0.0), 2),
                    'plate_score': round(float(p['score']), 2),
                    'plate_crop_base64': cv2_to_base64(plate_crop),
                    'plate_bbox': [px1, py1, px2, py2],
                    'vehicle_bbox': None
                })

        elapsed = round(time.time() - start_time, 2)
        return {
            'success': True,
            'inference_time_seconds': elapsed,
            'vehicles_found': len(vehicles),
            'plates_found': len([r for r in results_list if r['has_plate']]),
            'detections': results_list,
            'annotated_image_base64': cv2_to_base64(annotated, quality=85)
        }

    def process_video(self, video_path, output_path, progress_callback=None, max_frames=300):
        """
        Processes traffic video:
        - Tracks EVERY vehicle on screen.
        - Shows vehicle status and consensus plate text.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        mot_tracker = Sort()
        all_tracked_vehicles = {}  # tid -> {vehicle_type, first_seen, best_crop_b64}
        car_readings = defaultdict(list)
        active_consensus = {}

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            # 1. Detect ALL vehicles (conf=0.15 for tracking inclusiveness)
            v_res = self.coco_model(frame, conf=0.15, verbose=False)[0]
            detections_ = []
            for d in v_res.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = d
                if int(class_id) in self.vehicle_classes:
                    detections_.append([x1, y1, x2, y2, score])

            # 2. Update tracker
            track_ids = mot_tracker.update(np.asarray(detections_)) if len(detections_) > 0 else np.empty((0, 5))

            # 3. Detect plates
            p_res = self.plate_model(frame, conf=0.15, verbose=False)[0]
            for p in p_res.boxes.data.tolist():
                px1, py1, px2, py2, p_score, _ = p
                car_x1, car_y1, car_x2, car_y2, car_id = util.get_car(p, track_ids)
                if car_id != -1:
                    car_id = int(car_id)
                    py1_c, py2_c = max(0, int(py1)), min(height, int(py2))
                    px1_c, px2_c = max(0, int(px1)), min(width, int(px2))
                    crop_w = px2_c - px1_c
                    crop_h = py2_c - py1_c

                    if crop_w > 20 and crop_h > 10:
                        crop = frame[py1_c:py2_c, px1_c:px2_c]
                        text, ocr_score = util.read_license_plate(crop)
                        if text:
                            is_valid = util.is_valid_indian_plate(text)
                            car_readings[car_id].append({
                                'text': text,
                                'ocr_score': float(ocr_score or 0.0),
                                'is_valid': is_valid,
                                'crop_area': crop_w * crop_h,
                                'crop_b64': cv2_to_base64(crop),
                                'p_score': float(p_score)
                            })

                            # Calculate multi-frame consensus
                            votes = defaultdict(float)
                            best_crops = {}
                            for cand in car_readings[car_id]:
                                t = cand['text']
                                weight = cand['ocr_score'] * (2.0 if cand['is_valid'] else 1.0) * (1.0 + cand['crop_area'] / 4000.0)
                                votes[t] += weight
                                if t not in best_crops or cand['crop_area'] > best_crops[t]['crop_area']:
                                    best_crops[t] = cand

                            winning_text = max(votes.keys(), key=lambda k: votes[k])
                            win_cand = best_crops[winning_text]
                            active_consensus[car_id] = {
                                'car_id': car_id,
                                'plate_number': winning_text,
                                'state': get_state_from_plate(winning_text),
                                'ocr_score': round(win_cand['ocr_score'], 2),
                                'plate_score': round(win_cand['p_score'], 2),
                                'plate_crop_base64': win_cand['crop_b64']
                            }

            # 4. Annotate ALL vehicles on frame
            for track in track_ids:
                tx1, ty1, tx2, ty2, tid = track
                tid = int(tid)
                tx1_c, ty1_c = max(0, int(tx1)), max(0, int(ty1))
                tx2_c, ty2_c = min(width, int(tx2)), min(height, int(ty2))

                if tid not in all_tracked_vehicles and tx2_c > tx1_c and ty2_c > ty1_c:
                    v_crop = frame[ty1_c:ty2_c, tx1_c:tx2_c]
                    all_tracked_vehicles[tid] = {
                        'car_id': tid,
                        'vehicle_type': 'Vehicle',
                        'vehicle_crop_base64': cv2_to_base64(v_crop),
                        'first_seen': frame_idx
                    }

                draw_styled_box(frame, tx1, ty1, tx2, ty2, color=(59, 130, 246), label=f"Vehicle #{tid}")

                if tid in active_consensus:
                    p_info = active_consensus[tid]
                    banner_text = f"IND | {p_info['plate_number']}"
                    (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    bx = max(0, int(tx1))
                    by = max(th + 15, int(ty1) - 10)
                    cv2.rectangle(frame, (bx, by - th - 12), (bx + tw + 20, by), (255, 255, 255), -1)
                    cv2.rectangle(frame, (bx, by - th - 12), (bx + tw + 20, by), (0, 0, 0), 2)
                    cv2.rectangle(frame, (bx, by - th - 12), (bx + 45, by), (180, 50, 0), -1)
                    cv2.putText(frame, "IND", (bx + 6, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, p_info['plate_number'], (bx + 52, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1
            if progress_callback and frame_idx % 15 == 0:
                progress_callback(frame_idx, frames_to_process)

        cap.release()
        out.release()

        # Build comprehensive results list for EVERY vehicle tracked
        final_video_detections = []
        for tid, vinfo in all_tracked_vehicles.items():
            if tid in active_consensus:
                final_video_detections.append({
                    'car_id': tid,
                    'vehicle_type': vinfo['vehicle_type'],
                    'vehicle_crop_base64': vinfo['vehicle_crop_base64'],
                    'has_plate': True,
                    'plate_number': active_consensus[tid]['plate_number'],
                    'state': active_consensus[tid]['state'],
                    'ocr_score': active_consensus[tid]['ocr_score'],
                    'plate_score': active_consensus[tid]['plate_score'],
                    'plate_crop_base64': active_consensus[tid]['plate_crop_base64']
                })
            else:
                final_video_detections.append({
                    'car_id': tid,
                    'vehicle_type': vinfo['vehicle_type'],
                    'vehicle_crop_base64': vinfo['vehicle_crop_base64'],
                    'has_plate': False,
                    'plate_number': "Plate Not In View",
                    'state': "-",
                    'ocr_score': 0.0,
                    'plate_score': 0.0,
                    'plate_crop_base64': ""
                })

        return {
            'frames_processed': frame_idx,
            'unique_vehicles_tracked': len(final_video_detections),
            'detections': final_video_detections
        }
