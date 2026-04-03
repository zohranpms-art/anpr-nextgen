import cv2
import threading
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request
from flask_cors import CORS
import time
import os
import pyodbc
import base64
from datetime import datetime, timezone, timedelta
from PIL import Image
import io
import re
#app.py
import torch
from concurrent.futures import ThreadPoolExecutor
_ocr_executor = ThreadPoolExecutor(max_workers=1)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
EAT = timezone(timedelta(hours=3))
def now_eat():
    return datetime.now(EAT)

ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.onnx")
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.25
NMS_IOU = 0.45
_onnx_session = None

def load_onnx():
    global _onnx_session
    if _onnx_session is not None:
        return _onnx_session
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _onnx_session = ort.InferenceSession(ONNX_PATH, sess_options=opts,
                                              providers=['DmlExecutionProvider','CPUExecutionProvider'])
        print(" -> ONNX model loaded OK")
    except Exception as e:
        print(f" !! ONNX load failed: {e}")
        _onnx_session = None
    return _onnx_session

def _preprocess(pil_img):
    img = pil_img.resize((INPUT_W, INPUT_H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0)

def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def _nms(boxes, scores, iou_thresh):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept = []
    while order:
        i = order.pop(0)
        kept.append(i)
        order = [j for j in order if _iou(boxes[i], boxes[j]) < iou_thresh]
    return kept

def _parse_output(output, orig_w, orig_h, conf_thresh):
    out = output[0][0]
    scale_x = orig_w / INPUT_W
    scale_y = orig_h / INPUT_H
    cx, cy, w, h, conf = out[0], out[1], out[2], out[3], out[4]
    mask = conf >= conf_thresh
    if not mask.any():
        return []
    cx, cy, w, h, conf = cx[mask], cy[mask], w[mask], h[mask], conf[mask]
    x1 = np.clip(((cx - w/2)*scale_x).astype(int), 0, orig_w-1)
    y1 = np.clip(((cy - h/2)*scale_y).astype(int), 0, orig_h-1)
    x2 = np.clip(((cx + w/2)*scale_x).astype(int), 0, orig_w-1)
    y2 = np.clip(((cy + h/2)*scale_y).astype(int), 0, orig_h-1)
    valid = (x2 > x1) & (y2 > y1)
    if not valid.any():
        return []
    boxes = [[int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])] for i in np.where(valid)[0]]
    scores = [float(conf[i]) for i in np.where(valid)[0]]
    kept = _nms(boxes, scores, NMS_IOU)
    return [(boxes[i], scores[i]) for i in kept]

def detect_plate(image_bytes, save_crop_path):
    session = load_onnx()
    if session is None:
        return None, None
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = pil_img.size
        raw_out = session.run(None, {session.get_inputs()[0].name: _preprocess(pil_img)})
        detections = _parse_output(raw_out, orig_w, orig_h, CONF_THRESH)
        if not detections:
            return None, None
        detections.sort(key=lambda d: d[1], reverse=True)
        (x1, y1, x2, y2), conf = detections[0]
        pw, ph = x2-x1, y2-y1
        x1 = max(0, x1 - int(pw*0.05)); y1 = max(0, y1 - int(ph*0.10))
        x2 = min(orig_w, x2 + int(pw*0.15)); y2 = min(orig_h, y2 + int(ph*0.10))
        pil_img.crop((x1, y1, x2, y2)).save(save_crop_path, "JPEG", quality=90)
        conf_pct = round(conf*100, 1)
        print(f"[ONNX] Plate conf={conf_pct}% -> {os.path.basename(save_crop_path)}")
        return save_crop_path, conf_pct
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, None

def clean_ocr_text(raw_text):
    if not raw_text:
        return ""
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.strip().upper())
    ke_pat = re.compile(r'^[A-Z]{3}\d{3}[A-Z]$')
    if len(cleaned) == 8 and cleaned[0] == 'F':
        cand = cleaned[1:]
        if ke_pat.match(cand):
            cleaned = cand
    return cleaned

def get_plate_type(plate_text):
    if not plate_text or plate_text.strip().upper() in ("K.", "K"):
        return "No Plate"
    if re.match(r'^[A-Z]{3}\d{3}[A-Z]$', plate_text.strip().upper()):
        return "KE Plate"
    return "Others"

os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_onednn_kernel_skip_batch_norm_pattern'] = '1'
os.environ['FLAGS_use_mkldnn_int8'] = '0'
os.environ['PADDLE_USE_ONEDNN'] = '0'
_OCR_MODEL = None 

def load_ocr():
    global _OCR_MODEL
    if _OCR_MODEL is not None:
        return _OCR_MODEL
    try:
        from paddleocr import PaddleOCR
        _OCR_MODEL = PaddleOCR(lang='en', enable_mkldnn=True, cpu_threads=4, rec_batch_num=6, show_log=False)
        print(" -> PaddleOCR loaded OK")
    except Exception as e:
        print(f" !! PaddleOCR load failed: {e}")
    return _OCR_MODEL

def run_ocr(crop_path): 
    if not os.path.exists(crop_path):
        return ""
    ocr = load_ocr()
    if ocr is None:
        return ""
    try:
        result = ocr.ocr(crop_path)
        if not result:
            return ""

        def extract_texts(obj):
            texts = []
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    if (isinstance(item, (list, tuple)) and len(item) == 2
                            and isinstance(item[0], str)
                            and isinstance(item[1], float)):
                        if item[1] > 0.3:
                            texts.append(item[0].strip().upper())
                    else:
                        texts.extend(extract_texts(item))
            return texts

        texts = extract_texts(result)
        return clean_ocr_text(" ".join(texts))
    except Exception as e:
        import traceback
        print("OCR CRASHED:", e)
        traceback.print_exc()
        return ""

app = Flask(__name__)
CORS(app)

import urllib.request as _ur
import json as _json

BRIDGE_URL = "http://localhost:8889"

def notify_led(cam_id: int, plate: str, device_type: str):
    if not plate or plate.strip().upper() in ("K.", "K", ""):
        return
    try:
        body = _json.dumps({
            "cam_id":      cam_id,
            "plate":       plate.strip().upper(),
            "device_type": device_type,
        }).encode()
        req = _ur.Request(
            f"{BRIDGE_URL}/led/display",
            data    = body,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        with _ur.urlopen(req, timeout=2) as resp:
            pass
    except Exception as _e:
        print(f"[LED] notify_led cam={cam_id} plate={plate} — bridge error: {_e}")

DB_SERVER = r"AMAAN-PMS\SQLEXPRESS01"
DB_NAME = "AMAAN_PMS"

DRIVER_CANDIDATES = [
    "ODBC Driver 17 for SQL Server",
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 13 for SQL Server",
    "SQL Server Native Client 11.0",
    "SQL Server",
]

def _pick_driver():
    available = pyodbc.drivers()
    for d in DRIVER_CANDIDATES:
        if d in available:
            return d
    return available[0] if available else "SQL Server"

_DRIVER = _pick_driver()
print(f" -> Using ODBC driver: {_DRIVER}")

DB_CONN_STR = (
    f"Driver={{{_DRIVER}}};"
    f"Server={DB_SERVER};"
    f"Database={DB_NAME};"
    f"Trusted_Connection=yes;"
)

def get_db():
    return pyodbc.connect(DB_CONN_STR)

def _add_col_if_missing(cur, table, col, defn):
    cur.execute("""SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME=? AND COLUMN_NAME=?""", table, col)
    if cur.fetchone()[0] == 0:
        cur.execute(f"ALTER TABLE [AMAAN_PMS].[dbo].[{table}] ADD [{col}] {defn}")
        print(f" -> Added [{col}] to {table}")
    else:
        print(f" -> [{col}] already exists in {table}")

def _ensure_all_cols():
    try:
        conn = get_db(); cur = conn.cursor()
        for col, defn in [
            ("IsCorrect",  "BIT NOT NULL DEFAULT 1"),
            ("IsModified", "BIT NOT NULL DEFAULT 0"),
            ("UpdatedAt",  "DATETIME NULL"),
        ]:
            _add_col_if_missing(cur, "db_tbl_27_ANPR_dump", col, defn)
        conn.commit(); conn.close()
        print(" -> dump table columns ready")
    except Exception as e:
        print(f" !! dump col setup failed: {e}")

SNAPSHOT_BASE = os.path.join(os.path.expanduser("~"), "Documents", "ANPR_Snapshots")
os.makedirs(SNAPSHOT_BASE, exist_ok=True)

def get_today_dir():
    today = now_eat().strftime("%d-%b-%Y")
    base = os.path.join(SNAPSHOT_BASE, today)
    for sub in ("large", "crop"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    return base

CAMERAS = [
    {"id": 0, "name": "Camera 1", "location": "Entry 1", "device_type": "Entry1", "url": "rtsp://10.10.10.151:8557", "snap_url": "http://10.10.10.151/snapshot/last_ivs_result.jpg"},
    {"id": 1, "name": "Camera 2", "location": "Entry 2", "device_type": "Entry2", "url": "rtsp://10.10.10.152:8557", "snap_url": "http://10.10.10.152/snapshot/last_ivs_result.jpg"},
    {"id": 2, "name": "Camera 3", "location": "Exit 1",  "device_type": "Exit1",  "url": "rtsp://10.10.10.153:8557", "snap_url": "http://10.10.10.153/snapshot/last_ivs_result.jpg"},
    {"id": 3, "name": "Camera 4", "location": "Exit 2",  "device_type": "Exit2",  "url": "rtsp://10.10.10.154:8557", "snap_url": "http://10.10.10.154/snapshot/last_ivs_result.jpg"},
]

frames = {cam["id"]: None for cam in CAMERAS}
locks  = {cam["id"]: threading.Lock() for cam in CAMERAS}

def capture_frames(cam):
    cam_id, cam_url = cam["id"], cam["url"]
    if not cam_url:
        print(f"[SKIP] {cam['name']} - no URL"); return
    while True:
        cap = cv2.VideoCapture(cam_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            time.sleep(5); continue
        print(f"[OK] {cam['name']} connected")
        skip = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            skip += 1
            if skip % 3 != 0: continue
            with locks[cam_id]:
                frames[cam_id] = frame
            time.sleep(0.04)  # caps each camera to ~8fps
        cap.release(); time.sleep(3)

def generate(cam_id):
    while True:
        with locks[cam_id]:
            frame = frames[cam_id]
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype="uint8")
            cv2.putText(frame, "Connecting...", (190, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60,60,60), 2)
        try:
            if frame is None or frame.size == 0: continue
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret: continue
        except:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.1)

@app.route("/amaan.ico")
def serve_logo():
    from flask import send_from_directory
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "amaan.ico")

@app.route("/proxy_snap/<int:cam_id>")
def proxy_snap(cam_id):
    import urllib.request
    cam = next((c for c in CAMERAS if c["id"] == cam_id), None)
    if not cam or not cam["snap_url"]:
        from flask import Response as FR
        return FR(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9', mimetype="image/jpeg")
    try:
        url = cam["snap_url"] + f"?t={int(time.time())}"
        with urllib.request.urlopen(url, timeout=4) as resp:
            data = resp.read()
            ctype = resp.headers.get("Content-Type", "image/jpeg")
        from flask import Response as FR
        return FR(data, mimetype=ctype, headers={"Cache-Control":"no-store","Access-Control-Allow-Origin":"*"})
    except Exception as e:
        print(f"[PROXY ERR] cam {cam_id}: {e}")
        from flask import abort; abort(502)

def _time_to_str(t):
    if t is None: return None
    if hasattr(t, 'hour'): return f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}"
    s = str(t); return s[:8] if len(s) >= 8 else s

def _handle_entry(cur, dump_id, date_str, time_str, now_dt, device_type,
                  large_path, small_path, crop_path, ocr_text, plate_type, plate_conf):
    plate_key = (ocr_text or "").strip().upper()
    if not plate_key or plate_key in ("K.", "K", ""):
        return
    cur.execute("DELETE FROM [AMAAN_PMS].[dbo].[db_tbl_28_ANPR_entered] WHERE car_plate_no = ?", plate_key)
    cur.execute("""
        INSERT INTO [AMAAN_PMS].[dbo].[db_tbl_28_ANPR_entered]
            (car_plate_no, entry_time, action_status, charge_status, isValidated, created_on)
        VALUES (?, ?, ?, ?, ?, ?)
    """, plate_key, now_dt, 'active', 'unpaid', 0, now_dt)
    print(f"[ENTERED-DB] '{plate_key}' -> db_tbl_28_ANPR_entered")

def _handle_exit(cur, dump_id, date_str, time_str, now_dt, device_type,
                 large_path, small_path, crop_path, ocr_text, plate_type, plate_conf):
    plate_key = (ocr_text or "").strip().upper()
    entered_id = None
    entry_time_dt = None
    entered_time_str = None
    stayed_minutes = None
    if plate_key and plate_key not in ("K.", "K", ""):
        cur.execute("""
            SELECT TOP 1 Id, entry_time FROM [AMAAN_PMS].[dbo].[db_tbl_28_ANPR_entered]
            WHERE car_plate_no = ? ORDER BY entry_time DESC
        """, plate_key)
        row = cur.fetchone()
        if row:
            entered_id    = row[0]
            entry_time_dt = row[1]
            entered_time_str = _time_to_str(entry_time_dt)
            try:
                if entered_time_str:
                    ep = entered_time_str.split(":")
                    xp = time_str.split(":")
                    es = int(ep[0])*3600 + int(ep[1])*60 + int(float(ep[2] if len(ep)>2 else 0))
                    xs = int(xp[0])*3600 + int(xp[1])*60 + int(float(xp[2] if len(xp)>2 else 0))
                    diff = xs - es
                    if diff < 0: diff += 86400
                    stayed_minutes = diff // 60
            except Exception as e:
                print(f"[STAY CALC ERR] {e}")
            cur.execute("DELETE FROM [AMAAN_PMS].[dbo].[db_tbl_28_ANPR_entered] WHERE Id = ?", entered_id)
            print(f"[EXIT-DB] '{plate_key}' removed from entered, stayed ~{stayed_minutes}m")
        else:
            print(f"[EXIT-DB] '{plate_key}' not found in entered - logging exit only")
    cur.execute("""
        INSERT INTO [AMAAN_PMS].[dbo].[db_tbl_29_ANPR_exited]
            (car_plate_no, entry_time, exit_time, park_time_total, action_status, charge_status, isValidated, created_on)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, plate_key or None, entry_time_dt, now_dt, stayed_minutes, 'exited', 'unpaid', 0, now_dt)
    print(f"[EXITED-DB] '{plate_key}' -> db_tbl_29_ANPR_exited")
    return entered_time_str

@app.route("/api/save_snapshot", methods=["POST"])
def api_save_snapshot():
    try:
        data        = request.get_json()
        cam_id      = int(data["cam_id"])
        device_type = data["device_type"]
        img_bytes   = base64.b64decode(data["image_b64"])
        now         = now_eat()
        date_str    = now.strftime("%Y-%m-%d")
        time_str    = now.strftime("%H:%M:%S")
        ts_str      = now.strftime("%Y%m%d_%H%M%S_%f")
        now_dt      = now.replace(tzinfo=None)
        save_dir    = get_today_dir()
        date_folder = os.path.basename(save_dir)
        crop_name   = f"crop_{device_type}_{ts_str}.jpg"
        crop_path   = os.path.join(save_dir, "crop", crop_name)
        detected_crop_path, plate_conf = detect_plate(img_bytes, crop_path)
        if not detected_crop_path or (plate_conf and plate_conf < 30):
            ocr_text = "K."; plate_type = "No Plate"; plate_conf = None
        else:
            future = _ocr_executor.submit(run_ocr, detected_crop_path)
            try:
                ocr_text = future.result(timeout=15)
            except Exception:
                ocr_text = ""
            plate_type = get_plate_type(ocr_text)
            print(f"[OCR] '{ocr_text}' Type: {plate_type}")
        large_name = f"large_{device_type}_{ts_str}.jpg"
        large_path = os.path.join(save_dir, "large", large_name)
        with open(large_path, "wb") as f:
            f.write(img_bytes)
        record_id = None
        entered_time_str = None
        try:
            conn = get_db(); cur = conn.cursor()
            if ocr_text and ocr_text != "K.":
                cur.execute("""
                    SELECT TOP 1 Id FROM [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump]
                    WHERE PlateNumber=? AND DeviceType=?
                      AND CreatedAt >= DATEADD(SECOND,-60,GETDATE()) ORDER BY Id DESC
                """, ocr_text, device_type)
                dup = cur.fetchone()
                if dup:
                    conn.close()
                    print(f"[DEDUP] Skipped '{ocr_text}' on {device_type}")
                    return jsonify({"ok":True,"skipped":True,"reason":"duplicate_plate","id":dup[0]})
            for attempt in range(3):
                try:
                    cur.execute("""
                    INSERT INTO [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump]
                        (CaptureDate, CaptureTime, CaptureDateTime, DeviceType,
                        LargeImagePath, SmallImagePath, CroppedPlatePath,
                        PlateNumber, ModifiedPlate, PlateType, Confidence, CreatedAt, IsCorrect, IsModified)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, date_str, time_str, now_dt, device_type,
                        large_path, None, detected_crop_path or None,
                        ocr_text or None, None, plate_type,
                        float(plate_conf) if plate_conf else None, now_dt, 1, 0)
                    cur.execute("SELECT @@IDENTITY")
                    row = cur.fetchone()
                    record_id = int(row[0]) if row and row[0] is not None else None
                    break
                except pyodbc.Error as ex:
                    if ex.args[0] == '40001':
                        print(f"[DEADLOCK] Retry {attempt+1}/3"); time.sleep(0.2*(attempt+1)); continue
                    raise
            is_entry = device_type.lower().startswith("entry")
            if is_entry:
                _handle_entry(cur, record_id, date_str, time_str, now_dt,
                              device_type, large_path, None, detected_crop_path,
                              ocr_text, plate_type, plate_conf)
            else:
                entered_time_str = _handle_exit(cur, record_id, date_str, time_str, now_dt,
                                                device_type, large_path, None, detected_crop_path,
                                                ocr_text, plate_type, plate_conf)
            conn.commit(); conn.close()
            print(f"[DB] Inserted dump id={record_id}")
            notify_led(cam_id, ocr_text or "", device_type)
        except Exception as e:
            import traceback; traceback.print_exc()
        return jsonify({
            "ok": True, "id": record_id, "date_folder": date_folder,
            "large_name": large_name, "small_name": None, "crop_name": crop_name,
            "plate_conf": plate_conf, "ocr_text": ocr_text, "plate_type": plate_type,
            "date": date_str, "time": time_str, "device": device_type,
            "entered_time": entered_time_str,
        })
    except Exception as e:
        import traceback; print(f"[SAVE ERR] {traceback.format_exc()}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/update_plate", methods=["POST"])
def api_update_plate():
    try:
        data = request.get_json()
        record_id = int(data["id"]); new_text = data["text"].strip().upper()
        new_type  = get_plate_type(new_text)
        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            UPDATE [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump]
            SET ModifiedPlate=?, PlateType=?, IsCorrect=0, IsModified=1, UpdatedAt=? WHERE Id=?
        """, new_text, new_type, now_eat().replace(tzinfo=None), record_id)
        conn.commit(); conn.close()
        return jsonify({"ok": True, "plate_type": new_type})
    except Exception as e:
        import traceback; print(f"[UPDATE ERR] {traceback.format_exc()}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/mark_correct", methods=["POST"])
def api_mark_correct():
    conn = None
    try:
        data = request.get_json()
        record_id  = int(data["id"]); is_correct = bool(data.get("is_correct", True))
        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            UPDATE [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump] SET IsCorrect=?, UpdatedAt=? WHERE Id=?
        """, 1 if is_correct else 0, now_eat().replace(tzinfo=None), record_id)
        conn.commit(); conn.close()
        return jsonify({"ok": True, "is_correct": is_correct})
    except Exception as e:
        import traceback; print(f"[CORRECT ERR] {traceback.format_exc()}")
        if conn:
            try: conn.close()
            except: pass
        return jsonify({"ok": False, "error": str(e)}), 500

# ── NEW: Retry OCR endpoint ───────────────────────────────────────────────────
@app.route("/api/retry_ocr", methods=["POST"])
def api_retry_ocr():
    """Re-run OCR on the existing cropped plate image for a given record ID."""
    conn = None
    try:
        data = request.get_json()
        record_id = int(data["id"])
        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            SELECT CroppedPlatePath, PlateNumber FROM [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump]
            WHERE Id=?
        """, record_id)
        row = cur.fetchone()
        if not row:
            conn.close()
            return jsonify({"ok": False, "error": "Record not found"}), 404
        crop_path, old_plate = row
        if not crop_path or not os.path.exists(crop_path):
            conn.close()
            return jsonify({"ok": False, "error": "Crop image not found on disk"}), 404
        # Re-run OCR
        new_ocr = run_ocr(crop_path)
        new_type = get_plate_type(new_ocr)
        print(f"[RETRY OCR] id={record_id} old='{old_plate}' new='{new_ocr}' type={new_type}")
        # Update DB - set PlateNumber to new result (not ModifiedPlate, this is a fresh OCR)
        cur.execute("""
            UPDATE [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump]
            SET PlateNumber=?, PlateType=?, UpdatedAt=? WHERE Id=?
        """, new_ocr or None, new_type, now_eat().replace(tzinfo=None), record_id)
        conn.commit(); conn.close()
        return jsonify({"ok": True, "id": record_id, "ocr_text": new_ocr, "plate_type": new_type})
    except Exception as e:
        import traceback; print(f"[RETRY OCR ERR] {traceback.format_exc()}")
        if conn:
            try: conn.close()
            except: pass
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/reports", methods=["GET"])
def api_reports():
    import traceback as tb
    conn = None
    try:
        date_from=request.args.get("from",""); date_to=request.args.get("to","")
        search=request.args.get("search","").strip().upper(); filter_mode=request.args.get("filter","")
        limit=request.args.get("limit","100")
        # Multi-value device filter (comma-separated)
        device_filter_raw=request.args.get("device","")
        device_list=[d.strip() for d in device_filter_raw.split(",") if d.strip()] if device_filter_raw else []
        # Multi-value plate type filter (comma-separated)
        plate_type_raw=request.args.get("plate_type","")
        plate_type_list=[p.strip() for p in plate_type_raw.split(",") if p.strip()] if plate_type_raw else []

        conn=get_db(); cur=conn.cursor()
        def col_exists(name):
            cur.execute("""SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME='db_tbl_27_ANPR_dump' AND COLUMN_NAME=?""",name)
            return cur.fetchone()[0]>0
        has_correct=col_exists("IsCorrect"); has_modified=col_exists("IsModified")
        if not has_correct:
            try: cur.execute("ALTER TABLE [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump] ADD IsCorrect BIT NOT NULL DEFAULT 1"); conn.commit(); has_correct=True
            except: pass
        if not has_modified:
            try: cur.execute("ALTER TABLE [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump] ADD IsModified BIT NOT NULL DEFAULT 0"); conn.commit(); has_modified=True
            except: pass
        where_parts,params=[],[]
        if date_from: where_parts.append("CaptureDate >= ?"); params.append(date_from)
        if date_to:   where_parts.append("CaptureDate <= ?"); params.append(date_to)
        if search:
            where_parts.append("(PlateNumber LIKE ? OR ModifiedPlate LIKE ? OR DeviceType LIKE ?)"); params+=[f"%{search}%"]*3
        # Device filter (multi)
        if device_list:
            placeholders=",".join(["?"]*len(device_list))
            where_parts.append(f"DeviceType IN ({placeholders})")
            params+=device_list
        # Plate type filter (multi)
        if plate_type_list:
            placeholders=",".join(["?"]*len(plate_type_list))
            where_parts.append(f"COALESCE(PlateType,'Others') IN ({placeholders})")
            params+=plate_type_list
        if filter_mode=="modified" and has_modified:   where_parts.append("IsModified = 1")
        elif filter_mode=="modified":                   where_parts.append("ModifiedPlate IS NOT NULL")
        elif filter_mode=="correct" and has_correct:    where_parts.append("IsCorrect = 1")
        where_sql=("WHERE "+" AND ".join(where_parts)) if where_parts else ""
        cur.execute(f"SELECT COUNT(DISTINCT COALESCE(ModifiedPlate,PlateNumber,'')) FROM [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump] {where_sql}",params)
        unique_count=cur.fetchone()[0] or 0
        if limit in ("all",""):  top_sql=""
        else:
            try: top_sql=f"TOP {int(limit)}"
            except: top_sql="TOP 100"
        cur.execute(f"""
            SELECT {top_sql} Id,CaptureDate,CaptureTime,DeviceType,
                LargeImagePath,SmallImagePath,CroppedPlatePath,
                PlateNumber,ModifiedPlate,COALESCE(PlateType,'Others') AS PlateType,
                Confidence,COALESCE(IsCorrect,1) AS IsCorrect,COALESCE(IsModified,0) AS IsModified
            FROM [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump] {where_sql} ORDER BY Id DESC
        """,params)
        rows=cur.fetchall(); conn.close(); conn=None
        def path_to_url(p,sub):
            if not p: return None
            try:
                fname=os.path.basename(p)
                for part in p.replace("\\","/").split("/"):
                    if re.match(r'\d{2}-[A-Za-z]{3}-\d{4}',part):
                        return f"/snapshots/{part}/{sub}/{fname}"
            except: pass
            return None
        records=[]
        for r in rows:
            rid,cap_date,cap_time,device,large_path,small_path,crop_path,plate,modified,plate_type,conf,isc_raw,ism_raw=r
            p=(plate or "").strip(); m=(modified or "").strip(); ism=bool(ism_raw); isc=bool(isc_raw)
            crop_url=path_to_url(crop_path,"crop")
            records.append({"id":rid,"date":str(cap_date) if cap_date else "","time":str(cap_time) if cap_time else "",
                "device":device or "","large_url":path_to_url(large_path,"large"),"crop_url":crop_url,
                "large_path":large_path or "","crop_path":crop_path or "",
                "plate":p,"modified":m if ism else "","display_plate":m if (ism and m) else p,
                "plate_type":plate_type or "","conf":round(float(conf),1) if conf else None,
                "is_correct":isc,"was_modified":ism})
        return jsonify({"ok":True,"total":len(records),"unique":unique_count,
                        "records":records,"date_from":date_from,"date_to":date_to})
    except Exception as e:
        err_detail=tb.format_exc(); print(f"[REPORTS ERR]\n{err_detail}")
        if conn:
            try: conn.close()
            except: pass
        return jsonify({"ok":False,"error":str(e),"detail":err_detail,"records":[],"total":0,"unique":0}),500

@app.route("/api/export_csv", methods=["GET"])
def api_export_csv():
    try:
        date_from=request.args.get("from",""); date_to=request.args.get("to","")
        search=request.args.get("search","").strip().upper()
        conn=get_db(); cur=conn.cursor()
        where_parts,params=[],[]
        if date_from: where_parts.append("CaptureDate >= ?"); params.append(date_from)
        if date_to:   where_parts.append("CaptureDate <= ?"); params.append(date_to)
        if search:
            where_parts.append("(PlateNumber LIKE ? OR ModifiedPlate LIKE ? OR DeviceType LIKE ?)"); params+=[f"%{search}%"]*3
        where_sql=("WHERE "+" AND ".join(where_parts)) if where_parts else ""
        cur.execute(f"""
            SELECT Id,CaptureDate,CaptureTime,DeviceType,PlateNumber,ModifiedPlate,PlateType,Confidence,
                   LargeImagePath,SmallImagePath,CroppedPlatePath
            FROM [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump] {where_sql} ORDER BY Id DESC
        """,params)
        rows=cur.fetchall(); conn.close()
        lines=["ID,Date,Time,Device,OCR Text,Modified Text,Plate Type,Confidence%,Large Path,Small Path,Crop Path"]
        for r in rows:
            rid,cd,ct,dev,pl,mo,pt,conf,lp,sp,cp=r
            def esc(v): return f'"{str(v or "").replace(chr(34),chr(39))}"'
            lines.append(f"{rid},{esc(cd)},{esc(ct)},{esc(dev)},{esc(pl)},{esc(mo)},{esc(pt)},{round(float(conf),1) if conf else ''},{esc(lp)},{esc(sp)},{esc(cp)}")
        from flask import Response as FR
        return FR("\n".join(lines),mimetype="text/csv",
                  headers={"Content-Disposition":f"attachment; filename=ANPR_{date_from or 'all'}_{date_to or 'all'}.csv"})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500

@app.route("/api/export_pdf", methods=["GET"])
def api_export_pdf():
    try:
        from reportlab.lib.pagesizes import A4,landscape
        from reportlab.platypus import SimpleDocTemplate,Table,TableStyle,Paragraph,Image as RLImage,HRFlowable
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        import io as _io
        date_from=request.args.get("from",""); date_to=request.args.get("to","")
        search=request.args.get("search","").strip().upper()
        conn=get_db(); cur=conn.cursor()
        where_parts,params=[],[]
        if date_from: where_parts.append("CaptureDate >= ?"); params.append(date_from)
        if date_to:   where_parts.append("CaptureDate <= ?"); params.append(date_to)
        if search:
            where_parts.append("(PlateNumber LIKE ? OR ModifiedPlate LIKE ? OR DeviceType LIKE ?)"); params+=[f"%{search}%"]*3
        where_sql=("WHERE "+" AND ".join(where_parts)) if where_parts else ""
        limit_param=request.args.get("limit","100")
        top_sql="" if limit_param=="all" else f"TOP {int(limit_param) if limit_param.isdigit() else 100}"
        cur.execute(f"""
            SELECT {top_sql} Id,CaptureDate,CaptureTime,DeviceType,LargeImagePath,CroppedPlatePath,
                PlateNumber,ModifiedPlate,COALESCE(PlateType,'Others') AS PlateType,Confidence
            FROM [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump] {where_sql} ORDER BY Id DESC
        """,params)
        rows=cur.fetchall(); conn.close()
        buf=_io.BytesIO()
        doc=SimpleDocTemplate(buf,pagesize=landscape(A4),leftMargin=10*mm,rightMargin=10*mm,topMargin=12*mm,bottomMargin=12*mm)
        cs=ParagraphStyle('c',fontSize=8,fontName='Helvetica',textColor=colors.HexColor('#1a1a2e'),leading=11)
        ps=ParagraphStyle('p',fontSize=9,fontName='Helvetica-Bold',textColor=colors.HexColor('#1e40af'))
        ms=ParagraphStyle('m',fontSize=9,fontName='Helvetica-Bold',textColor=colors.HexColor('#059669'))
        ts=ParagraphStyle('t',fontSize=16,fontName='Helvetica-Bold',textColor=colors.HexColor('#1a1a2e'),spaceAfter=4)
        ss=ParagraphStyle('s',fontSize=9,fontName='Helvetica',textColor=colors.HexColor('#6b7280'),spaceAfter=10)
        VEH_W=52*mm; CROP_W=38*mm; ROW_H=36*mm
        col_widths=[12*mm,20*mm,16*mm,18*mm,VEH_W,CROP_W,26*mm,26*mm,18*mm,14*mm]
        header=[Paragraph(f'<b>{h}</b>',cs) for h in ['ID','Date','Time','Device','Vehicle','Plate','OCR Text','Modified Text','Type','Conf%']]
        data=[header]
        def _li(path,mw,mh):
            if not path or not os.path.exists(path): return Paragraph('<font color="#9ca3af">No Image</font>',cs)
            try:
                img=RLImage(path); iw,ih=img.imageWidth,img.imageHeight; sc=min(mw/iw,mh/ih,1.0)
                img.drawWidth=iw*sc; img.drawHeight=ih*sc; return img
            except: return Paragraph('<font color="#9ca3af">Error</font>',cs)
        for r in rows:
            rid,cd,ct,device,lp,cp,plate,mod,pt,conf=r
            data.append([Paragraph(str(rid),cs),Paragraph(str(cd)[:10] if cd else "",cs),Paragraph(str(ct)[:8] if ct else "",cs),
                Paragraph(device or "",cs),_li(lp,VEH_W-2*mm,ROW_H-2*mm),_li(cp,CROP_W-2*mm,ROW_H*0.55),
                Paragraph((plate or "").strip(),ps) if plate else Paragraph("—",cs),
                Paragraph((mod or "").strip(),ms) if mod else Paragraph("—",cs),
                Paragraph(pt or "—",cs),Paragraph(f"{round(float(conf),1)}%" if conf else "—",cs)])
        tbl=Table(data,colWidths=col_widths,repeatRows=1)
        tbl.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1a1a2e')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),8),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f3ff')]),
            ('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#d0d5dd')),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('ALIGN',(0,1),(3,-1),'LEFT'),('ALIGN',(6,1),(9,-1),'LEFT'),
            ('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),
            ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
            ('ROWHEIGHT',(0,1),(-1,-1),ROW_H),('ROWHEIGHT',(0,0),(-1,0),8*mm),
        ]))
        story=[Paragraph("AMAAN ANPR - Detection Report",ts),
               Paragraph(f"Generated: {now_eat().strftime('%d %b %Y %H:%M:%S')} | Records: {len(rows)}",ss),
               HRFlowable(width="100%",thickness=1.5,color=colors.HexColor('#7c3aed'),spaceAfter=8),tbl]
        doc.build(story); buf.seek(0)
        from flask import Response as FR
        return FR(buf.read(),mimetype="application/pdf",
                  headers={"Content-Disposition":f"attachment; filename=ANPR_{date_from or 'all'}_{date_to or 'all'}.pdf"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok":False,"error":str(e)}),500

@app.route("/snapshots/<date_folder>/<subfolder>/<filename>")
def serve_snapshot(date_folder,subfolder,filename):
    from flask import send_from_directory
    return send_from_directory(os.path.join(SNAPSHOT_BASE,date_folder,subfolder),filename)

@app.route("/api/today_count")
def api_today_count():
    try:
        today = now_eat().strftime("%Y-%m-%d")
        today_start = now_eat().strftime("%Y-%m-%d") + " 00:00:00"
        today_end   = now_eat().strftime("%Y-%m-%d") + " 23:59:59"
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump] WHERE CaptureDate=?", today)
        count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump] WHERE CaptureDate=? AND DeviceType IN ('Entry1','Entry2')", today)
        entered = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump] WHERE CaptureDate=? AND DeviceType IN ('Exit1','Exit2')", today)
        exited = cur.fetchone()[0]
        try:
            cur.execute("SELECT COUNT(*) FROM [AMAAN_PMS].[dbo].[db_tbl_28_ANPR_entered] WHERE entry_time >= ? AND entry_time <= ?", today_start, today_end)
            inside = cur.fetchone()[0]
        except:
            inside = max(0, entered - exited)
        conn.close()
        return jsonify({"ok": True, "count": count, "entered": entered, "exited": exited, "inside": inside})
    except Exception as e:
        return jsonify({"ok": False, "count": 0, "entered": 0, "exited": 0, "inside": 0, "error": str(e)}), 500

@app.route("/api/inside_cars")
def api_inside_cars():
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT car_plate_no, entry_time, created_on FROM [AMAAN_PMS].[dbo].[db_tbl_28_ANPR_entered] ORDER BY entry_time DESC")
        rows = cur.fetchall(); conn.close()
        records = []
        for r in rows:
            plate, entry_time, created_on = r
            et = str(entry_time) if entry_time else ""
            entry_date = et[:10] if len(et) >= 10 else ""
            entry_time_str = et[11:19] if len(et) >= 19 else ""
            try:
                now = now_eat().replace(tzinfo=None)
                if entry_time:
                    diff = int((now - entry_time).total_seconds())
                    if diff < 0: diff = 0
                    hrs = diff // 3600; mins = (diff % 3600) // 60
                    duration = (f"{hrs}h {mins}m") if hrs > 0 else f"{mins}m"
                else:
                    duration = "-"
            except:
                duration = "-"
            records.append({"plate": plate or "", "entry_date": entry_date, "entry_time": entry_time_str, "duration": duration})
        return jsonify({"ok": True, "records": records, "total": len(records)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "records": [], "total": 0, "error": str(e)}), 500

@app.route("/api/ghost_exits")
def api_ghost_exits():
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            SELECT TOP 50 car_plate_no, exit_time, created_on
            FROM [AMAAN_PMS].[dbo].[db_tbl_29_ANPR_exited]
            WHERE entry_time IS NULL AND car_plate_no IS NOT NULL AND car_plate_no NOT IN ('K.', 'K', '')
            ORDER BY exit_time DESC
        """)
        rows = cur.fetchall(); conn.close()
        records = []
        for r in rows:
            plate, exit_time, created_on = r
            et = str(exit_time) if exit_time else ""
            exit_date = et[:10] if len(et) >= 10 else ""
            exit_time_str = et[11:19] if len(et) >= 19 else ""
            records.append({"plate": plate or "", "exit_date": exit_date, "exit_time": exit_time_str})
        return jsonify({"ok": True, "records": records, "total": len(records)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "records": [], "total": 0, "error": str(e)}), 500

@app.route("/api/plate_filter")
def api_plate_filter():
    try:
        plate_type  = request.args.get("plate_type", "KE Plate")
        date_from   = request.args.get("from", "")
        date_to     = request.args.get("to", "")
        limit_param = request.args.get("limit", "50")
        conn = get_db(); cur = conn.cursor()
        where_parts = ["COALESCE(PlateType,'Others') = ?"]
        params = [plate_type]
        if date_from: where_parts.append("CaptureDate >= ?"); params.append(date_from)
        if date_to:   where_parts.append("CaptureDate <= ?"); params.append(date_to)
        where_sql = "WHERE " + " AND ".join(where_parts)
        top_sql = "" if limit_param == "all" else f"TOP {int(limit_param) if str(limit_param).isdigit() else 200}"
        cur.execute(f"""
            SELECT {top_sql} Id, CaptureDate, CaptureTime, DeviceType,
                   LargeImagePath, CroppedPlatePath, PlateNumber, ModifiedPlate, PlateType, Confidence,
                   COALESCE(IsModified,0) AS IsModified
            FROM [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump] {where_sql} ORDER BY Id DESC
        """, params)
        rows = cur.fetchall(); conn.close()
        records = []
        for r in rows:
            rid, cap_date, cap_time, device, large_path, crop_path, plate, modified, pt, conf, ism = r
            def path_to_name(p): return os.path.basename(p) if p else None
            def get_folder(p):
                if not p: return None
                for part in p.replace("\\","/").split("/"):
                    if re.match(r'\d{2}-[A-Za-z]{3}-\d{4}', part): return part
                return None
            records.append({"id":rid,"date":str(cap_date)[:10] if cap_date else "","time":str(cap_time)[:8] if cap_time else "",
                "device":device or "","largeName":path_to_name(large_path),"cropName":path_to_name(crop_path),
                "dateFolder":get_folder(large_path),"plate":(plate or "").strip(),"modified":(modified or "").strip(),
                "plateType":pt or "Others","conf":round(float(conf),1) if conf else None,
                "was_modified":bool(ism)})
        return jsonify({"ok": True, "records": records, "total": len(records), "plate_type": plate_type})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "records": [], "total": 0, "error": str(e)}), 500

@app.route("/api/today_entered")
def api_today_entered():
    try:
        today = now_eat().strftime("%Y-%m-%d")
        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            SELECT TOP 50 Id, DeviceType, LargeImagePath, SmallImagePath, CroppedPlatePath,
                   PlateNumber, ModifiedPlate, PlateType, Confidence, CaptureDate, CaptureTime,
                   COALESCE(IsModified,0) AS IsModified
            FROM [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump]
            WHERE CaptureDate = ? AND DeviceType IN ('Entry1','Entry2') ORDER BY Id DESC
        """, today)
        rows = cur.fetchall(); conn.close()
        records = []
        for r in rows:
            rid, device, large_path, small_path, crop_path, plate, modified, plate_type, conf, cap_date, cap_time, ism = r
            def path_to_name(p): return os.path.basename(p) if p else None
            def get_folder(p):
                if not p: return None
                for part in p.replace("\\","/").split("/"):
                    if re.match(r'\d{2}-[A-Za-z]{3}-\d{4}', part): return part
                return None
            records.append({"id":rid,"device":device or "","largeName":path_to_name(large_path),"smallName":path_to_name(small_path),
                "cropName":path_to_name(crop_path),"dateFolder":get_folder(large_path),
                "plate":(plate or "").strip(),"modified":(modified or "").strip(),
                "plateType":plate_type or "Others","conf":round(float(conf),1) if conf else None,
                "date":str(cap_date)[:10] if cap_date else "","time":str(cap_time)[:8] if cap_time else "",
                "was_modified":bool(ism)})
        return jsonify({"ok": True, "records": records})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "records": [], "error": str(e)}), 500

@app.route("/api/dbtest")
def api_dbtest():
    try:
        conn=get_db(); cur=conn.cursor()
        cur.execute("SELECT COUNT(*) FROM [AMAAN_PMS].[dbo].[db_tbl_27_ANPR_dump]")
        count=cur.fetchone()[0]; conn.close()
        return jsonify({"ok":True,"driver":_DRIVER,"record_count":count})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500


_HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AMAAN ANPR</title>
<link rel="icon" type="image/x-icon" href="/amaan.ico">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --white:#fff;--bg:#f0f2f5;--border:#d0d5dd;
  --purple:#7c3aed;--purple-lt:#ede9fe;
  --blue-bg:#e8f4fd;--blue-txt:#1e6fa8;
  --text:#1a1a2e;--muted:#6b7280;--hover:#f5f3ff;
  --green:#16a34a;--th-bg:#f8f9fa;--th-bdr:#dee2e6;--dark:#1a1a2e;
}
html,body{height:100%;background:var(--bg);font-family:'Segoe UI',sans-serif;font-size:13px;color:var(--text);overflow:hidden}
.app{display:flex;flex-direction:column;height:100vh;overflow:hidden}
.titlebar{display:flex;align-items:center;justify-content:space-between;background:var(--white);border-bottom:1px solid var(--border);padding:0 14px;height:38px;flex-shrink:0}
.t-left{display:flex;align-items:center;gap:8px}
.t-title{font-size:13px;font-weight:700}
.winctrl{display:flex;gap:2px}
.winctrl span{width:28px;height:28px;display:grid;place-items:center;font-size:16px;color:var(--muted);cursor:pointer;border-radius:4px}
.winctrl span:hover{background:#f0f0f0}
.wc:hover{background:#e53e3e!important;color:white!important}
.statsbar{display:flex;align-items:center;gap:20px;padding:5px 16px;background:var(--blue-bg);border-bottom:1px solid #b3d4f0;flex-shrink:0}
.s-lbl{font-size:11px;color:var(--muted)}
.s-val{font-size:22px;font-weight:700;color:var(--blue-txt);line-height:1}
.s-sub{font-size:11px;color:var(--muted)}
.sdiv{width:1px;height:32px;background:#b3d4f0}
.live{margin-left:auto;display:flex;align-items:center;gap:6px;font-size:11px;font-weight:700;color:var(--green)}
.ldot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);animation:blink 1s step-end infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
.clk{font-size:13px;font-weight:700;letter-spacing:1px}
.main{display:flex;flex:1;overflow:hidden;min-height:0}
.panel-table{width:62%;flex-shrink:0;display:flex;flex-direction:column;border-right:2px solid var(--border);background:var(--white);overflow:hidden}
.panel-cameras{width:19%;flex-shrink:0;display:flex;flex-direction:column;border-right:2px solid var(--border);background:#f4f5f8;overflow:hidden}
.panel-snaps{width:19%;flex-shrink:0;display:flex;flex-direction:column;background:#f4f5f8;overflow:hidden}
.latest-section{flex:0 0 55%;display:flex;flex-direction:column;min-height:0}
.history-section{flex:1;display:flex;flex-direction:column;min-height:0;overflow:hidden}
.sec-hdr{font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);padding:6px 10px;background:var(--white);border-bottom:1px solid var(--border);flex-shrink:0}
.latest-section .sec-hdr{background:#f8f9ff;color:#1e40af}
.tab-bar{display:flex;border-bottom:1px solid var(--border);background:var(--white);flex-shrink:0;overflow-x:auto}
.tab-btn{flex:1;padding:8px 4px;border:none;background:none;font-weight:600;font-size:11px;color:var(--muted);cursor:pointer;transition:all .2s;white-space:nowrap;min-width:0}
.tab-btn.active{color:var(--purple);border-bottom:3px solid var(--purple);background:var(--hover)}
.tab-content{flex:1;min-height:0;overflow:hidden;display:flex;flex-direction:column}
.tab-content.hidden{display:none!important}
.tbl-wrap{flex:1;overflow-y:auto;overflow-x:auto;min-height:0}
.tbl-wrap::-webkit-scrollbar{width:5px}
.tbl-wrap::-webkit-scrollbar-thumb{background:#c0c0c0;border-radius:3px}
table{width:100%;border-collapse:collapse;min-width:600px}
thead{position:sticky;top:0;z-index:2}
th{background:var(--th-bg);border-bottom:2px solid var(--th-bdr);border-right:1px solid var(--th-bdr);padding:9px 10px;text-align:left;font-size:12px;font-weight:600;color:var(--muted);white-space:nowrap}
th:last-child{border-right:none}
td{border-bottom:1px solid #eef0f3;border-right:1px solid #eef0f3;padding:3px 10px;vertical-align:middle;font-size:12px}
td:last-child{border-right:none}
tr:hover td{background:var(--hover);cursor:pointer}
tr.sel td{background:var(--purple-lt)}
.ci{padding:4px 6px}
.thumb{width:90px;height:60px;background:#e5e7eb;border:1px solid #d1d5db;border-radius:3px;display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:10px;overflow:hidden}
.thumb-crop{width:130px;height:50px;background:#e5e7eb;border:1px solid #d1d5db;border-radius:3px;display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:10px;overflow:hidden}
.thumb img,.thumb-crop img{width:100%;height:100%;object-fit:contain;cursor:zoom-in}
.ocr{font-family:'Courier New',monospace;font-size:12px;font-weight:600;color:#1e40af}
.ocr-edit{font-family:'Courier New',monospace;font-size:12px;font-weight:600;color:#059669}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;background:var(--purple-lt);color:var(--purple)}
.badge.exit{background:#fef3c7;color:#92400e}
.pt-badge{display:inline-block;padding:2px 7px;border-radius:10px;font-size:10px;font-weight:700}
.pt-badge.ke{background:#dcfce7;color:#166534;border:1px solid #bbf7d0}
.pt-badge.other{background:#f3f4f6;color:#4b5563;border:1px solid #d1d5db}
.pt-badge.noplate{background:#fef2f2;color:#dc2626;border:1px solid #fecaca}
.no-data{text-align:center;padding:30px;color:var(--muted);font-style:italic}
.col-list{flex:1;min-height:0;overflow:hidden;padding:2px 4px;display:flex;flex-direction:column;gap:2px}
.cam-card{flex:1;min-height:0;background:var(--white);border:1px solid var(--border);border-radius:5px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.06);display:flex;flex-direction:column}
.cam-lbl{display:flex;align-items:center;justify-content:space-between;padding:2px 6px;background:var(--dark);color:white;font-size:10px;font-weight:600;flex-shrink:0}
.ltag{font-size:9px;background:var(--green);color:white;padding:1px 4px;border-radius:5px;animation:blink 1.5s step-end infinite}
.cam-feed{flex:1;min-height:0;background:#0a0a0f;position:relative;overflow:hidden}
.cam-feed img{width:100%;height:100%;object-fit:contain;display:block}
.cam-conn{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:6px;background:#0d0f1a;color:#4b5563;font-size:10px}
.spinner{width:16px;height:16px;border:2px solid #374151;border-top-color:var(--purple);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.snap-card{flex:1;min-height:0;background:var(--white);border:1px solid var(--border);border-radius:5px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.04);display:flex;flex-direction:column}
.snap-hdr{display:flex;align-items:center;justify-content:space-between;padding:2px 6px;background:var(--th-bg);border-bottom:1px solid var(--border);font-size:10px;font-weight:600;color:var(--text);flex-shrink:0}
.snap-status{display:flex;align-items:center;gap:4px}
.snap-dot{width:6px;height:6px;border-radius:50%;background:#9ca3af;transition:background .3s}
.snap-dot.active{background:var(--green);box-shadow:0 0 5px var(--green)}
.snap-ts{font-size:9px;color:var(--muted)}
.snap-body{flex:1;min-height:0;position:relative;background:#e5e7eb;overflow:hidden}
.snap-body img{width:100%;height:100%;object-fit:contain;display:block}
.snap-placeholder{width:100%;height:100%;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:4px;color:#9ca3af;font-size:10px}
.snap-icon{font-size:18px;opacity:.4}
@keyframes flash{0%{box-shadow:0 0 0 3px rgba(124,58,237,0)}30%{box-shadow:0 0 0 3px rgba(124,58,237,0.7)}100%{box-shadow:0 0 0 3px rgba(124,58,237,0)}}
.snap-card.updated{animation:flash .8s ease-out}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:100;align-items:center;justify-content:center}
.modal-overlay.show{display:flex}
.modal{background:var(--white);border-radius:10px;padding:24px;width:360px;box-shadow:0 8px 32px rgba(0,0,0,.25)}
.modal h3{font-size:15px;font-weight:700;margin-bottom:12px}
.modal input{width:100%;padding:9px 12px;border:1.5px solid var(--border);border-radius:6px;font-size:14px;font-family:'Courier New',monospace;letter-spacing:2px;text-transform:uppercase;margin-bottom:14px}
.modal input:focus{outline:none;border-color:var(--purple)}
.modal-btns{display:flex;gap:8px;justify-content:flex-end}
.lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:300;align-items:center;justify-content:center}
.lightbox.show{display:flex}
.lightbox img{max-width:92vw;max-height:90vh;object-fit:contain;border-radius:6px;box-shadow:0 8px 48px rgba(0,0,0,.6)}
.lb-close{position:fixed;top:16px;right:20px;width:36px;height:36px;background:rgba(255,255,255,.15);border:none;border-radius:50%;color:white;font-size:20px;cursor:pointer;display:flex;align-items:center;justify-content:center}
.lb-close:hover{background:rgba(255,255,255,.3)}
.lb-label{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);color:rgba(255,255,255,.7);font-size:11px;background:rgba(0,0,0,.5);padding:4px 12px;border-radius:20px;pointer-events:none}
.toolbar{display:flex;align-items:center;justify-content:center;gap:10px;padding:8px 16px;background:var(--white);border-top:2px solid var(--border);flex-shrink:0}
.btn{padding:7px 18px;border:1px solid var(--border);border-radius:5px;font-size:12px;font-weight:600;cursor:pointer;background:var(--white);color:var(--text)}
.btn:hover{background:var(--th-bg)}
.btn.primary{background:var(--purple);color:white;border-color:var(--purple)}
.btn.primary:hover{background:#6d28d9}
.btn.success{background:#16a34a;color:white;border-color:#16a34a}
.btn.success:hover{background:#15803d}
.btn.info{background:#0891b2;color:white;border-color:#0891b2}
.btn.info:hover{background:#0e7490}
/* ── Tab filter bar ── */
.tab-filter-bar{display:flex;align-items:center;gap:8px;padding:5px 10px;background:#f8f9ff;border-bottom:1px solid #e0e7ff;flex-shrink:0;flex-wrap:wrap}
.tab-filter-bar label{font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;white-space:nowrap}
.tab-filter-bar select{padding:3px 8px;border:1.5px solid var(--border);border-radius:5px;font-size:11px;font-weight:600;color:var(--text);background:var(--white);cursor:pointer;min-width:90px}
.tab-filter-bar select:focus{outline:none;border-color:var(--purple)}
.tab-filter-bar .filter-sep{width:1px;height:20px;background:#d0d5dd}
.tab-filter-bar .filter-count{font-size:11px;font-weight:700;color:var(--purple);margin-left:auto}
/* inline edit button inside table */
.inline-edit-btn{padding:2px 8px;border:1px solid #7c3aed;border-radius:4px;background:var(--purple-lt);color:var(--purple);font-size:10px;font-weight:700;cursor:pointer;white-space:nowrap}
.inline-edit-btn:hover{background:var(--purple);color:white}
/* OCR retry button */
.ocr-retry-btn{padding:1px 5px;border:1px solid #d97706;border-radius:4px;background:#fef3c7;color:#92400e;font-size:11px;cursor:pointer;white-space:nowrap;display:inline-flex;align-items:center;gap:2px;margin-left:4px;vertical-align:middle}
.ocr-retry-btn:hover{background:#d97706;color:white}
.ocr-retry-btn.spinning svg{animation:spin .7s linear infinite}
/* ── Multi-select dropdown ── */
.ms-wrap{position:relative;display:inline-block}
.ms-btn{display:flex;align-items:center;gap:5px;padding:4px 10px;border:1.5px solid var(--border);border-radius:5px;font-size:11px;font-weight:600;color:var(--text);background:var(--white);cursor:pointer;min-width:110px;justify-content:space-between;white-space:nowrap}
.ms-btn:hover,.ms-btn.open{border-color:var(--purple);color:var(--purple)}
.ms-btn .ms-arrow{font-size:9px;opacity:.6;transition:transform .2s}
.ms-btn.open .ms-arrow{transform:rotate(180deg)}
.ms-dropdown{display:none;position:absolute;top:calc(100% + 3px);left:0;min-width:160px;background:var(--white);border:1.5px solid var(--purple);border-radius:7px;box-shadow:0 4px 16px rgba(124,58,237,.15);z-index:50;padding:4px 0;max-height:220px;overflow-y:auto}
.ms-dropdown.open{display:block}
.ms-option{display:flex;align-items:center;gap:8px;padding:6px 12px;cursor:pointer;font-size:12px;font-weight:500;color:var(--text);transition:background .1s}
.ms-option:hover{background:var(--hover)}
.ms-option input[type=checkbox]{accent-color:var(--purple);width:14px;height:14px;cursor:pointer}
.ms-option.selected{color:var(--purple);font-weight:700}
.ms-all-btn{display:flex;align-items:center;justify-content:center;padding:5px 12px;border-top:1px solid var(--border);font-size:11px;font-weight:700;color:var(--muted);cursor:pointer;margin-top:2px}
.ms-all-btn:hover{color:var(--purple)}
/* Reports modal */
.rpt-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:150;align-items:flex-start;justify-content:center;padding:10px;overflow-y:auto}
.rpt-overlay.show{display:flex}
.rpt-modal{background:var(--white);border-radius:12px;width:100%;max-width:1400px;box-shadow:0 16px 64px rgba(0,0,0,.3);display:flex;flex-direction:column;margin:auto}
.rpt-header{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--border);flex-shrink:0}
.rpt-header h2{font-size:18px;font-weight:700;flex:1}
.rpt-close-btn{width:32px;height:32px;border:none;background:none;font-size:18px;cursor:pointer;color:var(--muted);border-radius:4px;display:flex;align-items:center;justify-content:center}
.rpt-close-btn:hover{background:#f0f0f0}
.rpt-body{flex:1;overflow-y:auto;padding:12px 20px}
.rpt-controls{display:flex;flex-wrap:wrap;align-items:flex-end;gap:12px;margin-bottom:12px;padding:10px 14px;background:#f8f9ff;border:1px solid #e0e7ff;border-radius:8px}
.rpt-ctrl-group{display:flex;flex-direction:column;gap:4px}
.rpt-ctrl-label{font-size:10px;font-weight:700;color:var(--muted);letter-spacing:1px;text-transform:uppercase}
.rpt-ctrl-sep{width:1px;height:40px;background:#d0d5dd;align-self:flex-end}
.quick-btns,.filter-btns,.action-btns,.limit-row,.search-row{display:flex;gap:6px;align-items:center}
.qbtn,.fbtn{padding:5px 12px;border-radius:5px;border:1.5px solid var(--border);font-size:12px;font-weight:600;cursor:pointer;background:var(--white);color:var(--text);white-space:nowrap}
.qbtn.today{border-color:#2563eb;color:#2563eb}.qbtn.today:hover,.qbtn.today.active{background:#2563eb;color:white}
.qbtn.yest{border-color:#059669;color:#059669}.qbtn.yest:hover,.qbtn.yest.active{background:#059669;color:white}
.qbtn.week{border-color:#d97706;color:#d97706}.qbtn.week:hover,.qbtn.week.active{background:#d97706;color:white}
.qbtn.month{border-color:#7c3aed;color:#7c3aed}.qbtn.month:hover,.qbtn.month.active{background:#7c3aed;color:white}
.qbtn.last30{border-color:#dc2626;color:#dc2626}.qbtn.last30:hover,.qbtn.last30.active{background:#dc2626;color:white}
.fbtn.f-modified:hover,.fbtn.f-modified.active{background:#f59e0b;color:white;border-color:#f59e0b}
.fbtn.f-correct:hover,.fbtn.f-correct.active{background:#16a34a;color:white;border-color:#16a34a}
.date-row{display:flex;align-items:center;gap:6px}
.date-row label{font-size:12px;font-weight:600;color:var(--muted);white-space:nowrap}
.date-row input[type=date]{padding:5px 8px;border:1.5px solid var(--border);border-radius:5px;font-size:12px}
.date-row input[type=date]:focus{outline:none;border-color:var(--purple)}
.limit-row input[type=number]{width:70px;padding:5px 8px;border:1.5px solid var(--border);border-radius:5px;font-size:12px;text-align:center}
.limit-row input:focus{outline:none;border-color:var(--purple)}
.all-btn{padding:5px 10px;border:1.5px solid #0891b2;border-radius:5px;font-size:12px;font-weight:700;cursor:pointer;background:var(--white);color:#0891b2}
.all-btn:hover,.all-btn.active{background:#0891b2;color:white}
.search-row input{padding:5px 10px;border:1.5px solid var(--border);border-radius:5px;font-size:12px;width:180px}
.search-row input:focus{outline:none;border-color:var(--purple)}
.rpt-stats{display:flex;gap:24px;padding:10px 16px;background:#f8f9ff;border:1px solid #e0e7ff;border-radius:8px;margin-bottom:12px;flex-wrap:wrap}
.rpt-stat{display:flex;flex-direction:column;gap:2px}
.rpt-stat-lbl{font-size:10px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.rpt-stat-val{font-size:20px;font-weight:700;color:var(--blue-txt);line-height:1.1}
.rpt-stat-val.green{color:#16a34a}
.rpt-stat-val.small{font-size:13px;font-weight:600;color:var(--text)}
.rpt-tbl-wrap{overflow-x:auto;border:1px solid var(--th-bdr);border-radius:6px}
.rpt-tbl-wrap table{min-width:950px;width:100%;border-collapse:collapse}
.rpt-tbl-wrap th{background:#f1f5f9;border-bottom:2px solid var(--th-bdr);border-right:1px solid var(--th-bdr);padding:8px 10px;font-size:11px;font-weight:700;color:var(--muted);white-space:nowrap;text-transform:uppercase;letter-spacing:.5px}
.rpt-tbl-wrap td{border-bottom:1px solid #eef0f3;border-right:1px solid #eef0f3;padding:4px 10px;font-size:12px;vertical-align:middle}
.rpt-tbl-wrap tr:last-child td{border-bottom:none}
.rpt-tbl-wrap tr:hover td{background:var(--hover)}
.rpt-loading{text-align:center;padding:40px;color:var(--muted);font-size:13px;display:flex;align-items:center;justify-content:center;gap:10px}
.rpt-welcome{text-align:center;padding:50px 20px;color:var(--muted)}
.rpt-welcome-icon{font-size:48px;margin-bottom:14px;opacity:.4}
.rpt-welcome h3{font-size:16px;font-weight:600;margin-bottom:8px;color:#374151}
.rpt-welcome p{font-size:13px}
.rpt-footer{display:flex;align-items:center;justify-content:space-between;padding:10px 20px;border-top:1px solid var(--border);font-size:12px;color:var(--muted);flex-shrink:0}
.rpt-thumb{width:80px;height:52px;background:#e5e7eb;border:1px solid #d1d5db;border-radius:3px;display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:10px;overflow:hidden}
.rpt-thumb img{width:100%;height:100%;object-fit:contain;cursor:zoom-in}
.rpt-thumb-crop{width:110px;height:40px;background:#e5e7eb;border:1px solid #d1d5db;border-radius:3px;display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:10px;overflow:hidden}
.rpt-thumb-crop img{width:100%;height:100%;object-fit:contain;cursor:zoom-in}
.correct-btn{width:28px;height:28px;border-radius:50%;border:2px solid #d1d5db;background:var(--white);cursor:pointer;font-size:14px;display:inline-flex;align-items:center;justify-content:center;transition:all .2s;color:#9ca3af}
.correct-btn:hover{border-color:#16a34a;color:#16a34a}
.correct-btn.checked{background:#16a34a;border-color:#16a34a;color:white}
.mod-yes{color:#059669;font-size:15px}
.mod-no{color:#d1d5db;font-size:15px}
.toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%) translateY(60px);background:#1a1a2e;color:white;padding:8px 20px;border-radius:20px;font-size:12px;font-weight:600;opacity:0;transition:all .3s;z-index:400;pointer-events:none}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
/* Reports inline edit modal - sits INSIDE the reports overlay z-stack */
.rpt-edit-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:220;align-items:center;justify-content:center}
.rpt-edit-overlay.show{display:flex}
</style>
</head>
<body>
<div class="app">

  <div class="titlebar">
    <div class="t-left">
      <img src="/amaan.ico" style="width:26px;height:26px;object-fit:contain;">
      <span class="t-title">AMAAN ANPR</span>
    </div>
    <div class="winctrl"><span>&ndash;</span><span>&#9633;</span><span class="wc">&times;</span></div>
  </div>

  <div class="statsbar">
    <div><div class="s-lbl">Today's Count</div><div class="s-val" id="stat-count">0</div></div>
    <div class="sdiv"></div>
    <div><div class="s-lbl">&#x2B06; Entered</div><div class="s-val" id="stat-entered" style="color:#16a34a">0</div></div>
    <div class="sdiv"></div>
    <div><div class="s-lbl">&#x2B07; Exited</div><div class="s-val" id="stat-exited" style="color:#dc2626">0</div></div>
    <div class="sdiv"></div>
    <div><div class="s-lbl">&#x1F697; Inside</div><div class="s-val" id="stat-inside" style="color:#7c3aed">0</div></div>
    <div class="sdiv"></div>
    <div><div class="s-lbl">Last Detection</div><div class="s-sub" id="stat-last">&mdash;</div></div>
    <div class="sdiv"></div>
    <div class="live"><div class="ldot"></div> LIVE</div>
    <div class="clk" id="clock">--:--:--</div>
  </div>

  <div class="main">

    <!-- LEFT: TABLE PANEL -->
    <div class="panel-table">

      <!-- TOP: Latest Detections -->
      <div class="latest-section">
        <div class="sec-hdr">Latest Detections</div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>Date</th><th>Time</th><th>Device</th><th>Large Image</th><th>Cropped Plate</th><th>OCR Text</th><th>Modified Text</th><th>Plate Type</th><th>Conf%</th></tr></thead>
            <tbody id="tbody"><tr><td class="no-data" colspan="9">Waiting for detections...</td></tr></tbody>
          </table>
        </div>
      </div>

      <!-- DRAG HANDLE -->
      <div id="drag-handle" style="height:6px;background:#e0e7ff;cursor:row-resize;flex-shrink:0;display:flex;align-items:center;justify-content:center;border-top:1px solid #c7d2fe;border-bottom:1px solid #c7d2fe;user-select:none;">
        <div style="width:40px;height:2px;background:#a5b4fc;border-radius:2px;pointer-events:none"></div>
      </div>

      <!-- BOTTOM: 5-Tab Section -->
      <div class="history-section">

        <div class="tab-bar">
          <button class="tab-btn active" data-tab="entered">&#x2B06; Entered</button>
          <button class="tab-btn" data-tab="exited">&#x2B07; Exited</button>
          <button class="tab-btn" data-tab="inside">&#x1F697; Inside</button>
          <button class="tab-btn" data-tab="ghostexit">&#x1F47B; Unreg. Exits</button>
          <button class="tab-btn" data-tab="platefilter">&#x1F6AB; Not Kenyan</button>
        </div>

        <!-- TAB: Entered -->
        <div class="tab-content" id="tab-entered">
          <div class="tab-filter-bar" id="filter-bar-entered">
            <label>Device:</label>
            <div class="ms-wrap" id="ms-entered-device">
              <button class="ms-btn" onclick="toggleMs('ms-entered-device')"><span class="ms-label">All Devices</span><span class="ms-arrow">&#9660;</span></button>
              <div class="ms-dropdown">
                <label class="ms-option"><input type="checkbox" value="Entry1">Entry 1</label>
                <label class="ms-option"><input type="checkbox" value="Entry2">Entry 2</label>
                <div class="ms-all-btn" onclick="clearMs('ms-entered-device')">Clear all</div>
              </div>
            </div>
            <div class="filter-sep"></div>
            <label>Plate Type:</label>
            <div class="ms-wrap" id="ms-entered-pt">
              <button class="ms-btn" onclick="toggleMs('ms-entered-pt')"><span class="ms-label">All Types</span><span class="ms-arrow">&#9660;</span></button>
              <div class="ms-dropdown">
                <label class="ms-option"><input type="checkbox" value="KE Plate">KE Plate</label>
                <label class="ms-option"><input type="checkbox" value="Others">Others</label>
                <label class="ms-option"><input type="checkbox" value="No Plate">No Plate</label>
                <div class="ms-all-btn" onclick="clearMs('ms-entered-pt')">Clear all</div>
              </div>
            </div>
            <span class="filter-count" id="filter-entered-count"></span>
          </div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>Date</th><th>Time</th><th>Device</th><th>Large Image</th><th>Cropped Plate</th><th>OCR Text</th><th>Modified Text</th><th>Plate Type</th><th>Conf%</th></tr></thead>
              <tbody id="tbody-entered"><tr><td class="no-data" colspan="9">No entered vehicles yet...</td></tr></tbody>
            </table>
          </div>
        </div>

        <!-- TAB: Exited -->
        <div class="tab-content hidden" id="tab-exited">
          <div class="tab-filter-bar" id="filter-bar-exited">
            <label>Device:</label>
            <div class="ms-wrap" id="ms-exited-device">
              <button class="ms-btn" onclick="toggleMs('ms-exited-device')"><span class="ms-label">All Devices</span><span class="ms-arrow">&#9660;</span></button>
              <div class="ms-dropdown">
                <label class="ms-option"><input type="checkbox" value="Exit1">Exit 1</label>
                <label class="ms-option"><input type="checkbox" value="Exit2">Exit 2</label>
                <div class="ms-all-btn" onclick="clearMs('ms-exited-device')">Clear all</div>
              </div>
            </div>
            <div class="filter-sep"></div>
            <label>Plate Type:</label>
            <div class="ms-wrap" id="ms-exited-pt">
              <button class="ms-btn" onclick="toggleMs('ms-exited-pt')"><span class="ms-label">All Types</span><span class="ms-arrow">&#9660;</span></button>
              <div class="ms-dropdown">
                <label class="ms-option"><input type="checkbox" value="KE Plate">KE Plate</label>
                <label class="ms-option"><input type="checkbox" value="Others">Others</label>
                <label class="ms-option"><input type="checkbox" value="No Plate">No Plate</label>
                <div class="ms-all-btn" onclick="clearMs('ms-exited-pt')">Clear all</div>
              </div>
            </div>
            <span class="filter-count" id="filter-exited-count"></span>
          </div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>Date</th><th>Entered Time</th><th>Exited Time</th><th>Stayed</th><th>Device</th><th>Large Image</th><th>Cropped Plate</th><th>OCR Text</th><th>Modified Text</th><th>Plate Type</th></tr></thead>
              <tbody id="tbody-exited"><tr><td class="no-data" colspan="10">No exited vehicles yet...</td></tr></tbody>
            </table>
          </div>
        </div>

        <!-- TAB: Inside -->
        <div class="tab-content hidden" id="tab-inside">
          <div style="display:flex;align-items:center;justify-content:space-between;padding:4px 10px;background:#f0fdf4;border-bottom:1px solid #bbf7d0;flex-shrink:0">
            <span style="font-size:11px;color:#16a34a;font-weight:700" id="inside-count-lbl">Loading...</span>
            <button class="btn" id="btn-refresh-inside" style="padding:3px 10px;font-size:11px;">&#x21bb; Refresh</button>
          </div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>Plate Number</th><th>Entry Date</th><th>Entry Time</th><th>Time Parked</th></tr></thead>
              <tbody id="tbody-inside"><tr><td class="no-data" colspan="4">Loading...</td></tr></tbody>
            </table>
          </div>
        </div>

        <!-- TAB: Unregistered Exits -->
        <div class="tab-content hidden" id="tab-ghostexit">
          <div style="display:flex;align-items:center;justify-content:space-between;padding:4px 10px;background:#fff7ed;border-bottom:1px solid #fed7aa;flex-shrink:0">
            <span style="font-size:11px;color:#ea580c;font-weight:700" id="ghost-count-lbl">Loading...</span>
            <button class="btn" id="btn-refresh-ghost" style="padding:3px 10px;font-size:11px;">&#x21bb; Refresh</button>
          </div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>Plate Number</th><th>Exit Date</th><th>Exit Time</th></tr></thead>
              <tbody id="tbody-ghostexit"><tr><td class="no-data" colspan="3">Loading...</td></tr></tbody>
            </table>
          </div>
        </div>

        <!-- TAB: Not Kenyan (Others only) -->
        <div class="tab-content hidden" id="tab-platefilter">
          <div class="tab-filter-bar" id="filter-bar-pf">
            <label>Device:</label>
            <div class="ms-wrap" id="ms-pf-device">
              <button class="ms-btn" onclick="toggleMs('ms-pf-device')"><span class="ms-label">All Devices</span><span class="ms-arrow">&#9660;</span></button>
              <div class="ms-dropdown">
                <label class="ms-option"><input type="checkbox" value="Entry1">Entry 1</label>
                <label class="ms-option"><input type="checkbox" value="Entry2">Entry 2</label>
                <label class="ms-option"><input type="checkbox" value="Exit1">Exit 1</label>
                <label class="ms-option"><input type="checkbox" value="Exit2">Exit 2</label>
                <div class="ms-all-btn" onclick="clearMs('ms-pf-device')">Clear all</div>
              </div>
            </div>
            <div class="filter-sep"></div>
            <label>Plate Type:</label>
            <div class="ms-wrap" id="ms-pf-pt">
              <button class="ms-btn" onclick="toggleMs('ms-pf-pt')"><span class="ms-label">Others</span><span class="ms-arrow">&#9660;</span></button>
              <div class="ms-dropdown">
                <label class="ms-option"><input type="checkbox" value="Others" checked>Others</label>
                <label class="ms-option"><input type="checkbox" value="No Plate">No Plate</label>
                <label class="ms-option"><input type="checkbox" value="KE Plate">KE Plate</label>
                <div class="ms-all-btn" onclick="clearMs('ms-pf-pt')">Clear all</div>
              </div>
            </div>
            <div class="filter-sep"></div>
            <button class="btn" id="btn-refresh-pf" style="padding:3px 10px;font-size:11px;">&#x21bb; Refresh</button>
            <span class="filter-count" id="pf-count-lbl">Loading...</span>
          </div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>Date</th><th>Time</th><th>Device</th><th>Large Image</th><th>Cropped Plate</th><th>OCR Text</th><th>Modified Text</th><th>Plate Type</th><th>Conf%</th><th>Edit</th></tr></thead>
              <tbody id="tbody-platefilter"><tr><td class="no-data" colspan="10">Loading...</td></tr></tbody>
            </table>
          </div>
        </div>

      </div><!-- /history-section -->
    </div><!-- /panel-table -->

    <!-- MIDDLE: Live Cameras -->
    <div class="panel-cameras">
      <div class="sec-hdr">&#127909; Live Cameras</div>
      <div class="col-list">
        <div class="cam-card"><div class="cam-lbl"><span>Cam 1 &mdash; Entry 1</span><span class="ltag">LIVE</span></div><div class="cam-feed"><img src="/video/0" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"><div class="cam-conn" style="display:none"><div class="spinner"></div><span>Connecting...</span></div></div></div>
        <div class="cam-card"><div class="cam-lbl"><span>Cam 2 &mdash; Entry 2</span><span class="ltag">LIVE</span></div><div class="cam-feed"><img src="/video/1" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"><div class="cam-conn" style="display:none"><div class="spinner"></div><span>Connecting...</span></div></div></div>
        <div class="cam-card"><div class="cam-lbl"><span>Cam 3 &mdash; Exit 1</span><span class="ltag">LIVE</span></div><div class="cam-feed"><img src="/video/2" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"><div class="cam-conn" style="display:none"><div class="spinner"></div><span>Connecting...</span></div></div></div>
        <div class="cam-card"><div class="cam-lbl"><span>Cam 4 &mdash; Exit 2</span><span class="ltag">LIVE</span></div><div class="cam-feed"><img src="/video/3" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"><div class="cam-conn" style="display:none"><div class="spinner"></div><span>Connecting...</span></div></div></div>
      </div>
    </div><!-- /panel-cameras -->

    <!-- RIGHT: Latest Snapshots -->
    <div class="panel-snaps">
      <div class="sec-hdr">&#128247; Latest Snapshots</div>
      <div class="col-list">
        <div class="snap-card" id="snap-card-0"><div class="snap-hdr"><span>Entry 1</span><div class="snap-status"><div class="snap-dot" id="snap-dot-0"></div><span class="snap-ts" id="snap-ts-0">Waiting...</span></div></div><div class="snap-body" id="snap-body-0"><div class="snap-placeholder"><span class="snap-icon">&#128247;</span><span>No detection yet</span></div></div></div>
        <div class="snap-card" id="snap-card-1"><div class="snap-hdr"><span>Entry 2</span><div class="snap-status"><div class="snap-dot" id="snap-dot-1"></div><span class="snap-ts" id="snap-ts-1">Waiting...</span></div></div><div class="snap-body" id="snap-body-1"><div class="snap-placeholder"><span class="snap-icon">&#128247;</span><span>No detection yet</span></div></div></div>
        <div class="snap-card" id="snap-card-2"><div class="snap-hdr"><span>Exit 1</span><div class="snap-status"><div class="snap-dot" id="snap-dot-2"></div><span class="snap-ts" id="snap-ts-2">Waiting...</span></div></div><div class="snap-body" id="snap-body-2"><div class="snap-placeholder"><span class="snap-icon">&#128247;</span><span>No detection yet</span></div></div></div>
        <div class="snap-card" id="snap-card-3"><div class="snap-hdr"><span>Exit 2</span><div class="snap-status"><div class="snap-dot" id="snap-dot-3"></div><span class="snap-ts" id="snap-ts-3">Not Configured</span></div></div><div class="snap-body" id="snap-body-3"><div class="snap-placeholder"><span class="snap-icon">&#128247;</span><span>No detection yet</span></div></div></div>
      </div>
    </div><!-- /panel-snaps -->

  </div><!-- /main -->

  <div class="toolbar">
    <button class="btn" id="btn-edit">Edit Selected Text</button>
    <button class="btn primary" id="btn-reports">&#128202; View Reports</button>
  </div>

</div><!-- /app -->

<div class="toast" id="toast"></div>

<!-- Main Edit Modal (for main table + Not Kenyan tab) -->
<div class="modal-overlay" id="edit-modal">
  <div class="modal">
    <h3>&#9999; Edit Plate Text</h3>
    <input id="edit-input" type="text" placeholder="Enter corrected plate..." maxlength="20">
    <div class="modal-btns"><button class="btn" id="edit-cancel">Cancel</button><button class="btn primary" id="edit-save">Save</button></div>
  </div>
</div>

<!-- Reports Edit Modal (z-index above reports overlay) -->
<div class="rpt-edit-overlay" id="rpt-edit-modal">
  <div class="modal">
    <h3>&#9999; Edit Plate Text</h3>
    <input id="rpt-edit-input" type="text" placeholder="Enter corrected plate..." maxlength="20">
    <div class="modal-btns"><button class="btn" id="rpt-edit-cancel">Cancel</button><button class="btn primary" id="rpt-edit-save">Save</button></div>
  </div>
</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox">
  <button class="lb-close" id="lb-close">&times;</button>
  <img id="lb-img" src="" alt="">
  <div class="lb-label" id="lb-label"></div>
</div>

<!-- Reports Modal -->
<div class="rpt-overlay" id="rpt-overlay">
  <div class="rpt-modal">
    <div class="rpt-header"><span style="font-size:22px">&#128202;</span><h2>ANPR Detailed Reports</h2><button class="rpt-close-btn" id="rpt-close">&times;</button></div>
    <div class="rpt-body">
      <div class="rpt-controls">
        <div class="rpt-ctrl-group"><div class="rpt-ctrl-label">Quick Reports</div><div class="quick-btns"><button class="qbtn today" data-q="today">Today</button><button class="qbtn yest" data-q="yesterday">Yesterday</button><button class="qbtn week" data-q="week">This Week</button><button class="qbtn month" data-q="month">This Month</button><button class="qbtn last30" data-q="last30">Last 30 Days</button></div></div>
        <div class="rpt-ctrl-sep"></div>
        <div class="rpt-ctrl-group"><div class="rpt-ctrl-label">Custom Range</div><div class="date-row"><label>From:</label><input type="date" id="rpt-from"><label>To:</label><input type="date" id="rpt-to"></div></div>
        <div class="rpt-ctrl-sep"></div>
        <!-- Multi-select Device filter for reports -->
        <div class="rpt-ctrl-group">
          <div class="rpt-ctrl-label">Device</div>
          <div class="ms-wrap" id="ms-rpt-device">
            <button class="ms-btn" onclick="toggleMs('ms-rpt-device')"><span class="ms-label">All Devices</span><span class="ms-arrow">&#9660;</span></button>
            <div class="ms-dropdown">
              <label class="ms-option"><input type="checkbox" value="Entry1">Entry 1</label>
              <label class="ms-option"><input type="checkbox" value="Entry2">Entry 2</label>
              <label class="ms-option"><input type="checkbox" value="Exit1">Exit 1</label>
              <label class="ms-option"><input type="checkbox" value="Exit2">Exit 2</label>
              <div class="ms-all-btn" onclick="clearMs('ms-rpt-device')">Clear all</div>
            </div>
          </div>
        </div>
        <div class="rpt-ctrl-sep"></div>
        <!-- Multi-select Plate Type filter for reports -->
        <div class="rpt-ctrl-group">
          <div class="rpt-ctrl-label">Plate Type</div>
          <div class="ms-wrap" id="ms-rpt-pt">
            <button class="ms-btn" onclick="toggleMs('ms-rpt-pt')"><span class="ms-label">All Types</span><span class="ms-arrow">&#9660;</span></button>
            <div class="ms-dropdown">
              <label class="ms-option"><input type="checkbox" value="KE Plate">KE Plate</label>
              <label class="ms-option"><input type="checkbox" value="Others">Others</label>
              <label class="ms-option"><input type="checkbox" value="No Plate">No Plate</label>
              <div class="ms-all-btn" onclick="clearMs('ms-rpt-pt')">Clear all</div>
            </div>
          </div>
        </div>
        <div class="rpt-ctrl-sep"></div>
        <div class="rpt-ctrl-group"><div class="rpt-ctrl-label">Filter</div><div class="filter-btns"><button class="fbtn f-modified" id="fbtn-modified">&#9999; Modified</button><button class="fbtn f-correct" id="fbtn-correct">&#10003; Correct</button></div></div>
        <div class="rpt-ctrl-sep"></div>
        <div class="rpt-ctrl-group"><div class="rpt-ctrl-label">Show Records</div><div class="limit-row"><input type="number" id="rpt-limit" value="100" min="1" max="99999"><button class="all-btn" id="rpt-all-btn">All</button></div></div>
        <div class="rpt-ctrl-sep"></div>
        <div class="rpt-ctrl-group"><div class="rpt-ctrl-label">Search</div><div class="search-row"><input type="text" id="rpt-search" placeholder="Plate, device..."><button class="btn" id="rpt-clear-search" style="padding:5px 10px;">&times;</button></div></div>
        <div class="rpt-ctrl-sep"></div>
        <div class="rpt-ctrl-group" id="pf-rpt-group" style="display:none"><div class="rpt-ctrl-label">Plate Filter Type</div>
          <select id="pf-rpt-select" style="padding:5px 10px;border:1.5px solid var(--purple);border-radius:5px;font-size:12px;font-weight:600;color:var(--purple);background:white;cursor:pointer;">
            <option value="KE Plate" selected>KE Plate</option>
            <option value="Others">Others</option>
            <option value="No Plate">No Plate</option>
          </select>
        </div>
        <div class="rpt-ctrl-group"><div class="rpt-ctrl-label">Actions</div><div class="action-btns"><button class="btn primary" id="rpt-generate">&#128200; Generate</button><button class="btn success" id="rpt-export-csv">&#128196; CSV</button><button class="btn info" id="rpt-export-excel">&#128202; Excel</button><button class="btn" id="rpt-export-pdf" style="background:#dc2626;color:white;border-color:#dc2626;">&#128196; PDF</button></div></div>
      </div>
      <div class="rpt-stats" id="rpt-stats" style="display:none">
        <div class="rpt-stat"><div class="rpt-stat-lbl">Total Records</div><div class="rpt-stat-val" id="rpt-total">0</div></div>
        <div class="rpt-stat"><div class="rpt-stat-lbl">Unique Plates</div><div class="rpt-stat-val green" id="rpt-unique">0</div></div>
        <div class="rpt-stat"><div class="rpt-stat-lbl">Date Range</div><div class="rpt-stat-val small" id="rpt-range">&mdash;</div></div>
      </div>
      <div id="rpt-welcome" class="rpt-welcome"><div class="rpt-welcome-icon">&#128202;</div><h3>Select a report to get started</h3><p>Choose a quick report or set a custom date range, then click <strong>Generate</strong>.</p></div>
      <div style="display:flex;gap:0;border-bottom:2px solid var(--border);margin-bottom:12px;overflow-x:auto">
        <button class="rpt-tab-btn active" data-rpt="all" style="padding:8px 18px;border:none;background:none;font-weight:600;font-size:12px;color:var(--purple);border-bottom:3px solid var(--purple);cursor:pointer;white-space:nowrap">All Detections</button>
        <button class="rpt-tab-btn" data-rpt="entered" style="padding:8px 18px;border:none;background:none;font-weight:600;font-size:12px;color:var(--muted);cursor:pointer;white-space:nowrap">&#x2B06; Entered</button>
        <button class="rpt-tab-btn" data-rpt="exited" style="padding:8px 18px;border:none;background:none;font-weight:600;font-size:12px;color:var(--muted);cursor:pointer;white-space:nowrap">&#x2B07; Exited</button>
        <button class="rpt-tab-btn" data-rpt="inside" style="padding:8px 18px;border:none;background:none;font-weight:600;font-size:12px;color:var(--muted);cursor:pointer;white-space:nowrap">&#x1F697; Still Inside</button>
        <button class="rpt-tab-btn" data-rpt="ghost" style="padding:8px 18px;border:none;background:none;font-weight:600;font-size:12px;color:var(--muted);cursor:pointer;white-space:nowrap">&#x1F47B; Unregistered Exits</button>
        <button class="rpt-tab-btn" data-rpt="platefilter" style="padding:8px 18px;border:none;background:none;font-weight:600;font-size:12px;color:var(--muted);cursor:pointer;white-space:nowrap">&#x1F50D; Plate Filter</button>
      </div>
      <div class="rpt-tbl-wrap" id="rpt-tbl-wrap" style="display:none"><table><thead><tr id="rpt-thead-tr"></tr></thead><tbody id="rpt-tbody"></tbody></table></div>
      <div id="rpt-loading" style="display:none" class="rpt-loading"><div class="spinner"></div>Loading records...</div>
      <div id="rpt-empty" style="display:none" class="rpt-loading"></div>
    </div>
    <div class="rpt-footer"><span id="rpt-showing"></span><button class="btn" id="rpt-close-bottom">&times; Close</button></div>
  </div>
</div>

<script>
// ── Utilities ────────────────────────────────────────────────────────────────
function eatNow(){return new Date(Date.now()+3*3600000);}
function eatTimeStr(){return eatNow().toISOString().slice(11,19);}
function eatDateStr(){
  var d=eatNow(),m=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][d.getUTCMonth()];
  return String(d.getUTCDate()).padStart(2,'0')+'/'+m+'/'+d.getUTCFullYear();
}
function isoToday(){
  var d=eatNow();
  return d.getUTCFullYear()+'-'+String(d.getUTCMonth()+1).padStart(2,'0')+'-'+String(d.getUTCDate()).padStart(2,'0');
}
function isoOffset(days){
  var d=new Date(eatNow().getTime()+days*86400000);
  return d.getUTCFullYear()+'-'+String(d.getUTCMonth()+1).padStart(2,'0')+'-'+String(d.getUTCDate()).padStart(2,'0');
}
function formatDate(iso){
  if(!iso)return'';
  try{var p=iso.split('-');return p[2]+'/'+['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][parseInt(p[1])-1]+'/'+p[0];}
  catch(e){return iso;}
}
function plateBadge(pt){
  if(!pt)return'';
  var cls=pt==='KE Plate'?'ke':pt==='No Plate'?'noplate':'other';
  return '<span class="pt-badge '+cls+'">'+pt+'</span>';
}
function calcStayed(enteredTime,exitedTime){
  if(!enteredTime||!exitedTime)return'\u2014';
  try{
    var ep=enteredTime.split(':'),xp=exitedTime.split(':');
    var es=parseInt(ep[0])*3600+parseInt(ep[1])*60+parseInt(ep[2]||0);
    var xs=parseInt(xp[0])*3600+parseInt(xp[1])*60+parseInt(xp[2]||0);
    var diff=xs-es;if(diff<0)diff+=86400;
    var hrs=Math.floor(diff/3600),mins=Math.floor((diff%3600)/60);
    return hrs>0?hrs+'h '+mins+'m':mins+'m';
  }catch(e){return'\u2014';}
}
var _toastTimer=null;
function showToast(msg,dur){
  var t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');
  clearTimeout(_toastTimer);_toastTimer=setTimeout(function(){t.classList.remove('show');},dur||2500);
}
function tick(){document.getElementById('clock').textContent=eatTimeStr();}
tick();setInterval(tick,1000);
setTimeout(function(){location.reload();},6*60*60*1000);

// ── Lightbox ─────────────────────────────────────────────────────────────────
var lb=document.getElementById('lightbox'),lbImg=document.getElementById('lb-img'),lbLbl=document.getElementById('lb-label');
function openLightbox(src,label){lbImg.src=src;lbLbl.textContent=label||'';lb.classList.add('show');}
function closeLightbox(){lb.classList.remove('show');lbImg.src='';}
document.getElementById('lb-close').addEventListener('click',closeLightbox);
lb.addEventListener('click',function(e){if(e.target===lb)closeLightbox();});
document.addEventListener('keydown',function(e){if(e.key==='Escape'){closeLightbox();closeReports();}});

// ── Multi-select dropdown helpers ─────────────────────────────────────────────
function toggleMs(wrapperId){
  var wrap=document.getElementById(wrapperId);
  var btn=wrap.querySelector('.ms-btn');
  var dd=wrap.querySelector('.ms-dropdown');
  var isOpen=dd.classList.contains('open');
  // Close all open dropdowns first
  document.querySelectorAll('.ms-dropdown.open').forEach(function(d){
    d.classList.remove('open');
    d.closest('.ms-wrap').querySelector('.ms-btn').classList.remove('open');
  });
  if(!isOpen){dd.classList.add('open');btn.classList.add('open');}
}
// Close dropdowns when clicking outside
document.addEventListener('click',function(e){
  if(!e.target.closest('.ms-wrap')){
    document.querySelectorAll('.ms-dropdown.open').forEach(function(d){
      d.classList.remove('open');
      d.closest('.ms-wrap').querySelector('.ms-btn').classList.remove('open');
    });
  }
});
function getMs(wrapperId){
  var wrap=document.getElementById(wrapperId);
  var checked=[];
  wrap.querySelectorAll('input[type=checkbox]:checked').forEach(function(cb){checked.push(cb.value);});
  return checked;
}
function clearMs(wrapperId){
  var wrap=document.getElementById(wrapperId);
  wrap.querySelectorAll('input[type=checkbox]').forEach(function(cb){cb.checked=false;});
  updateMsLabel(wrapperId);
  // Close dropdown
  wrap.querySelector('.ms-dropdown').classList.remove('open');
  wrap.querySelector('.ms-btn').classList.remove('open');
}
function updateMsLabel(wrapperId){
  var wrap=document.getElementById(wrapperId);
  var checked=getMs(wrapperId);
  var lbl=wrap.querySelector('.ms-label');
  if(!checked.length){
    // figure out placeholder from context
    var placeholder=wrapperId.indexOf('device')>=0?'All Devices':'All Types';
    // For pf-pt default shows "Others" preselected initially — handle separately
    lbl.textContent=placeholder;
    wrap.querySelector('.ms-btn').style.borderColor='';
    wrap.querySelector('.ms-btn').style.color='';
  } else {
    lbl.textContent=checked.length===1?checked[0]:checked.length+' selected';
    wrap.querySelector('.ms-btn').style.borderColor='var(--purple)';
    wrap.querySelector('.ms-btn').style.color='var(--purple)';
  }
}
// Wire change events on all multi-select checkboxes
document.querySelectorAll('.ms-wrap').forEach(function(wrap){
  wrap.querySelectorAll('input[type=checkbox]').forEach(function(cb){
    cb.addEventListener('change',function(){
      updateMsLabel(wrap.id);
      // Trigger filter updates
      if(wrap.id==='ms-entered-device'||wrap.id==='ms-entered-pt'){
        renderTabTable('tbody-entered',applyTabFilter('entered',enteredRows));
        updateTabCount('entered',enteredRows);
      }
      if(wrap.id==='ms-exited-device'||wrap.id==='ms-exited-pt'){
        renderTabTable('tbody-exited',applyTabFilter('exited',exitedRows));
        updateTabCount('exited',exitedRows);
      }
      if(wrap.id==='ms-pf-device'||wrap.id==='ms-pf-pt'){
        var filtered=applyTabFilter('pf',pfAllRows);
        renderPFTable(filtered);
        updateTabCount('pf',pfAllRows);
      }
    });
  });
});
// Initialize pf-pt with "Others" checked
(function(){
  var cb=document.querySelector('#ms-pf-pt input[value="Others"]');
  if(cb){cb.checked=true;updateMsLabel('ms-pf-pt');}
})();

// ── Data stores ───────────────────────────────────────────────────────────────
var MAX_LIVE=50,MAX_TAB=50;
var tableRows=[],enteredRows=[],exitedRows=[],selectedIdx=null,todayCount=0;
var pfAllRows=[];

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats(){
  document.getElementById('stat-count').textContent=todayCount;
  if(tableRows.length)document.getElementById('stat-last').textContent='Last: '+tableRows[0].time;
}
async function fetchTodayCount(){
  try{
    var r=await fetch('/api/today_count'),d=await r.json();
    if(d.ok){
      todayCount=d.count;
      document.getElementById('stat-entered').textContent=d.entered||0;
      document.getElementById('stat-exited').textContent=d.exited||0;
      document.getElementById('stat-inside').textContent=d.inside||0;
      updateStats();
    }
  }catch(e){}
}
fetchTodayCount();setInterval(fetchTodayCount,30000);

// ── Load today entered on startup ─────────────────────────────────────────────
async function loadTodayEntered(){
  try{
    var resp=await fetch('/api/today_entered');
    var data=await resp.json();
    if(!data.ok||!data.records.length)return;
    data.records.forEach(function(r){
      var row={id:r.id||null,date:formatDate(r.date),time:r.time,device:r.device,
        dateFolder:r.dateFolder,largeName:r.largeName,smallName:r.smallName,cropName:r.cropName,
        conf:r.conf,plate:r.plate||null,modified:r.modified||null,
        plateType:r.plateType,isModified:r.was_modified||false,isCorrect:true,enteredTime:null};
      var plateKey=(r.plate||'').trim().toUpperCase();
      if(plateKey&&plateKey!=='K.'&&plateKey!=='K'){
        var exists=enteredRows.some(function(e){return((e.plate||'').trim().toUpperCase())===plateKey;});
        if(!exists) enteredRows.push(row);
      }
    });
    renderTabTable('tbody-entered',applyTabFilter('entered',enteredRows));
    updateTabCount('entered',enteredRows);
  }catch(e){}
}
loadTodayEntered();

// ── OCR Retry SVG icon ────────────────────────────────────────────────────────
var RETRY_SVG='<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.7"/></svg>';

function ocrRetryBtn(rid,hasCrop){
  if(!hasCrop||!rid)return'';
  return '<button class="ocr-retry-btn" title="Retry OCR" data-id="'+rid+'" onclick="retryOcr(this)">'+RETRY_SVG+'</button>';
}

async function retryOcr(btn){
  var rid=parseInt(btn.dataset.id);
  btn.classList.add('spinning');
  btn.disabled=true;
  try{
    var resp=await fetch('/api/retry_ocr',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:rid})});
    var res=await resp.json();
    if(!res.ok){showToast('\u26a0 OCR failed: '+(res.error||'unknown'));return;}
    var newOcr=res.ocr_text||'';
    var newType=res.plate_type||'Others';
    showToast('\u21bb OCR: '+(newOcr||'(empty)')+' \u2192 '+newType);
    // Update all local row stores
    [tableRows,enteredRows,exitedRows,pfAllRows].forEach(function(arr){
      var r=arr.find(function(r){return r.id===rid;});
      if(r){r.plate=newOcr;r.plateType=newType;}
    });
    renderTable();
    renderTabTable('tbody-entered',applyTabFilter('entered',enteredRows));
    renderTabTable('tbody-exited',applyTabFilter('exited',exitedRows));
    renderPFTable(applyTabFilter('pf',pfAllRows));
    if(rptLoaded)fetchReports();
  }catch(e){showToast('\u26a0 Retry error: '+e.message);}
  finally{btn.classList.remove('spinning');btn.disabled=false;}
}

// ── Row HTML builders ─────────────────────────────────────────────────────────
function createRowHTML(r){
  var ls=(r.dateFolder&&r.largeName)?'/snapshots/'+r.dateFolder+'/large/'+r.largeName:'';
  var cs=(r.cropName&&r.dateFolder)?'/snapshots/'+r.dateFolder+'/crop/'+r.cropName:'';
  var lT=ls?'<div class="thumb"><img src="'+ls+'" data-src="'+ls+'" data-lbl="Large '+r.device+'" onerror="this.parentElement.textContent=\'No Image\'"></div>':'<div class="thumb">-</div>';
  var pT=cs?'<div class="thumb-crop"><img src="'+cs+'" data-src="'+cs+'" data-lbl="Plate '+(r.plate||'')+'" onerror="this.parentElement.textContent=\'No Plate\'"></div>':'<div class="thumb-crop" style="font-size:10px;color:#9ca3af">-</div>';
  var isExit=(r.device||'').toLowerCase().indexOf('exit')>=0;
  var badge='<span class="badge'+(isExit?' exit':'')+'">'+r.device+'</span>';
  var ocr=r.plate?'<span class="ocr">'+r.plate+'</span>'+ocrRetryBtn(r.id,!!cs):'<span style="color:#9ca3af">-</span>'+ocrRetryBtn(r.id,!!cs);
  var mod=r.modified?'<span class="ocr-edit">'+r.modified+'</span>':'<span style="color:#9ca3af">-</span>';
  var conf=r.conf!=null?r.conf+'%':'';
  return '<tr data-live="1"><td>'+r.date+'</td><td>'+r.time+'</td><td>'+badge+'</td><td class="ci">'+lT+'</td><td class="ci">'+pT+'</td><td>'+ocr+'</td><td>'+mod+'</td><td>'+plateBadge(r.plateType)+'</td><td>'+conf+'</td></tr>';
}

function createExitedRowHTML(r){
  var ls=(r.dateFolder&&r.largeName)?'/snapshots/'+r.dateFolder+'/large/'+r.largeName:'';
  var cs=(r.cropName&&r.dateFolder)?'/snapshots/'+r.dateFolder+'/crop/'+r.cropName:'';
  var lT=ls?'<div class="thumb"><img src="'+ls+'" data-src="'+ls+'" data-lbl="Exit Image" onerror="this.parentElement.textContent=\'No Image\'"></div>':'<div class="thumb">-</div>';
  var pT=cs?'<div class="thumb-crop"><img src="'+cs+'" data-src="'+cs+'" data-lbl="Plate '+(r.plate||'')+'" onerror="this.parentElement.textContent=\'No Plate\'"></div>':'<div class="thumb-crop" style="font-size:10px;color:#9ca3af">-</div>';
  var badge='<span class="badge exit">'+r.device+'</span>';
  var ocr=r.plate?'<span class="ocr">'+r.plate+'</span>'+ocrRetryBtn(r.id,!!cs):'<span style="color:#9ca3af">-</span>'+ocrRetryBtn(r.id,!!cs);
  var mod=r.modified?'<span class="ocr-edit">'+r.modified+'</span>':'<span style="color:#9ca3af">-</span>';
  var entT=r.enteredTime||''; var exT=r.time||'';
  var stayed=calcStayed(entT,exT);
  var entTd=entT?'<span style="color:#16a34a;font-weight:600">'+entT+'</span>':'<span style="color:#9ca3af">\u2014</span>';
  var exTd=exT?'<span style="color:#dc2626;font-weight:600">'+exT+'</span>':'<span style="color:#9ca3af">\u2014</span>';
  var stTd=stayed!=='\u2014'?'<span style="color:#7c3aed;font-weight:700">'+stayed+'</span>':'<span style="color:#9ca3af">\u2014</span>';
  return '<tr><td>'+r.date+'</td><td>'+entTd+'</td><td>'+exTd+'</td><td>'+stTd+'</td><td>'+badge+'</td><td class="ci">'+lT+'</td><td class="ci">'+pT+'</td><td>'+ocr+'</td><td>'+mod+'</td><td>'+plateBadge(r.plateType)+'</td></tr>';
}

function createPFRowHTML(r){
  var ls=(r.dateFolder&&r.largeName)?'/snapshots/'+r.dateFolder+'/large/'+r.largeName:'';
  var cs=(r.cropName&&r.dateFolder)?'/snapshots/'+r.dateFolder+'/crop/'+r.cropName:'';
  var lT=ls?'<div class="thumb"><img src="'+ls+'" data-src="'+ls+'" data-lbl="'+r.device+'" onerror="this.parentElement.textContent=\'No Image\'"></div>':'<div class="thumb">-</div>';
  var pT=cs?'<div class="thumb-crop"><img src="'+cs+'" data-src="'+cs+'" data-lbl="'+(r.plate||'')+'" onerror="this.parentElement.textContent=\'No Plate\'"></div>':'<div class="thumb-crop" style="color:#9ca3af;font-size:10px">-</div>';
  var isExit=(r.device||'').toLowerCase().indexOf('exit')>=0;
  var badge='<span class="badge'+(isExit?' exit':'')+'">'+r.device+'</span>';
  var ocr=r.plate?'<span class="ocr">'+r.plate+'</span>'+ocrRetryBtn(r.id,!!cs):'<span style="color:#9ca3af">-</span>'+ocrRetryBtn(r.id,!!cs);
  var mod=(r.modified&&r.was_modified)?'<span class="ocr-edit">'+r.modified+'</span>':'<span style="color:#9ca3af">-</span>';
  var conf=r.conf!=null?r.conf+'%':'';
  var editBtn=r.id?'<button class="inline-edit-btn" data-id="'+r.id+'" data-plate="'+((r.modified&&r.was_modified)?r.modified:r.plate||'')+'" onclick="openInlineEdit(this)">&#9999; Edit</button>':'<span style="color:#9ca3af;font-size:10px">-</span>';
  return '<tr><td>'+formatDate(r.date)+'</td><td>'+r.time+'</td><td>'+badge+'</td><td class="ci">'+lT+'</td><td class="ci">'+pT+'</td><td>'+ocr+'</td><td>'+mod+'</td><td>'+plateBadge(r.plateType)+'</td><td>'+conf+'</td><td style="text-align:center">'+editBtn+'</td></tr>';
}

// ── Tab filtering ─────────────────────────────────────────────────────────────
function applyTabFilter(tab, rows){
  if(tab==='entered'){
    var devs=getMs('ms-entered-device');
    var pts=getMs('ms-entered-pt');
    return rows.filter(function(r){
      if(devs.length&&devs.indexOf(r.device)<0)return false;
      if(pts.length&&pts.indexOf(r.plateType)<0)return false;
      return true;
    });
  }
  if(tab==='exited'){
    var devs=getMs('ms-exited-device');
    var pts=getMs('ms-exited-pt');
    return rows.filter(function(r){
      if(devs.length&&devs.indexOf(r.device)<0)return false;
      if(pts.length&&pts.indexOf(r.plateType)<0)return false;
      return true;
    });
  }
  if(tab==='pf'){
    var devs=getMs('ms-pf-device');
    var pts=getMs('ms-pf-pt');
    return rows.filter(function(r){
      if(devs.length&&devs.indexOf(r.device)<0)return false;
      if(pts.length&&pts.indexOf(r.plateType)<0)return false;
      return true;
    });
  }
  return rows;
}

function updateTabCount(tab, allRows){
  var filtered=applyTabFilter(tab,allRows);
  if(tab==='entered') document.getElementById('filter-entered-count').textContent=filtered.length+' record'+(filtered.length!==1?'s':'');
  if(tab==='exited')  document.getElementById('filter-exited-count').textContent=filtered.length+' record'+(filtered.length!==1?'s':'');
  if(tab==='pf')      document.getElementById('pf-count-lbl').textContent=filtered.length+' record'+(filtered.length!==1?'s':'');
}

// ── Render helpers ────────────────────────────────────────────────────────────
function renderTable(){
  var tb=document.getElementById('tbody');
  if(!tableRows.length){tb.innerHTML='<tr><td class="no-data" colspan="9">Waiting for detections...</td></tr>';return;}
  tb.innerHTML=tableRows.map(createRowHTML).join('');
  attachImageListeners();
  document.querySelectorAll('#tbody tr[data-live="1"]').forEach(function(row,idx){
    row.addEventListener('click',function(e){
      if(e.target.tagName==='IMG'||e.target.tagName==='BUTTON'||e.target.tagName==='SVG'||e.target.tagName==='POLYLINE'||e.target.tagName==='PATH')return;
      selectedIdx=idx;
      document.querySelectorAll('#tbody tr').forEach(function(r){r.classList.remove('sel');});
      row.classList.add('sel');
    });
  });
}

function renderTabTable(tbodyId,rows){
  var tb=document.getElementById(tbodyId);
  if(!rows.length){
    var cols=tbodyId==='tbody-exited'?10:9;
    var msg=tbodyId==='tbody-entered'?'No entered vehicles match filters.':'No exited vehicles match filters.';
    tb.innerHTML='<tr><td class="no-data" colspan="'+cols+'">'+msg+'</td></tr>';return;
  }
  if(tbodyId==='tbody-exited'){
    tb.innerHTML=rows.map(createExitedRowHTML).join('');
  }else{
    tb.innerHTML=rows.map(createRowHTML).join('');
  }
  attachImageListeners();
}

function renderPFTable(rows){
  var tb=document.getElementById('tbody-platefilter');
  if(!rows.length){tb.innerHTML='<tr><td class="no-data" colspan="10">No records match filters.</td></tr>';return;}
  tb.innerHTML=rows.map(createPFRowHTML).join('');
  attachImageListeners();
}

function attachImageListeners(){
  document.querySelectorAll('img[data-src]').forEach(function(img){
    if(img._lbAttached)return; img._lbAttached=true;
    img.style.cursor='zoom-in';
    img.addEventListener('click',function(e){e.stopPropagation();openLightbox(img.dataset.src,img.dataset.lbl);});
  });
}

// ── Inline edit from Not Kenyan tab (main-page modal) ────────────────────────
var _inlineEditId=null,_inlineEditSource=null;
function openInlineEdit(btn){
  _inlineEditId=parseInt(btn.dataset.id);
  _inlineEditSource='pf';
  var editInput=document.getElementById('edit-input');
  editInput.value=(btn.dataset.plate||'').trim().toUpperCase();
  document.getElementById('edit-modal').classList.add('show');
  editInput.focus();editInput.select();
}

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(function(btn){
  btn.addEventListener('click',function(){
    document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active');});
    btn.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(function(c){c.classList.add('hidden');});
    document.getElementById('tab-'+btn.dataset.tab).classList.remove('hidden');
    if(btn.dataset.tab==='inside') fetchInsideCars();
    if(btn.dataset.tab==='ghostexit') fetchGhostExits();
    if(btn.dataset.tab==='platefilter') fetchPlateFilter();
  });
});

// ── Inside Cars ───────────────────────────────────────────────────────────────
async function fetchInsideCars(){
  document.getElementById('tbody-inside').innerHTML='<tr><td class="no-data" colspan="4">Loading...</td></tr>';
  document.getElementById('inside-count-lbl').textContent='Loading...';
  try{
    var resp=await fetch('/api/inside_cars'); var data=await resp.json();
    if(!data.ok||!data.records.length){
      document.getElementById('tbody-inside').innerHTML='<tr><td class="no-data" colspan="4">No vehicles currently inside.</td></tr>';
      document.getElementById('inside-count-lbl').textContent='0 vehicles inside'; return;
    }
    document.getElementById('inside-count-lbl').textContent=data.total+' vehicle'+(data.total!==1?'s':'')+' currently inside';
    document.getElementById('tbody-inside').innerHTML=data.records.map(function(r){
      return '<tr><td><span class="ocr" style="font-size:14px;letter-spacing:2px">'+r.plate+'</span></td>'+
        '<td>'+formatDate(r.entry_date)+'</td>'+
        '<td><span style="color:#16a34a;font-weight:600">'+r.entry_time+'</span></td>'+
        '<td><span style="color:#7c3aed;font-weight:700">'+r.duration+'</span></td></tr>';
    }).join('');
  }catch(e){document.getElementById('tbody-inside').innerHTML='<tr><td class="no-data" colspan="4">Error: '+e.message+'</td></tr>';}
}
document.getElementById('btn-refresh-inside').addEventListener('click',fetchInsideCars);

// ── Ghost Exits ───────────────────────────────────────────────────────────────
async function fetchGhostExits(){
  document.getElementById('tbody-ghostexit').innerHTML='<tr><td class="no-data" colspan="3">Loading...</td></tr>';
  document.getElementById('ghost-count-lbl').textContent='Loading...';
  try{
    var resp=await fetch('/api/ghost_exits'); var data=await resp.json();
    if(!data.ok||!data.records.length){
      document.getElementById('tbody-ghostexit').innerHTML='<tr><td class="no-data" colspan="3">No unregistered exits found.</td></tr>';
      document.getElementById('ghost-count-lbl').textContent='0 unregistered exits'; return;
    }
    document.getElementById('ghost-count-lbl').textContent=data.total+' exit'+(data.total!==1?'s':'')+' with no entry record';
    document.getElementById('tbody-ghostexit').innerHTML=data.records.map(function(r){
      return '<tr><td><span class="ocr" style="font-size:14px;letter-spacing:2px">'+r.plate+'</span></td>'+
        '<td>'+formatDate(r.exit_date)+'</td>'+
        '<td><span style="color:#ea580c;font-weight:600">'+r.exit_time+'</span></td></tr>';
    }).join('');
  }catch(e){document.getElementById('tbody-ghostexit').innerHTML='<tr><td class="no-data" colspan="3">Error: '+e.message+'</td></tr>';}
}
document.getElementById('btn-refresh-ghost').addEventListener('click',fetchGhostExits);

// ── Plate Filter (Not Kenyan) ─────────────────────────────────────────────────
async function fetchPlateFilter(){
  pfAllRows=[];
  renderPFTable([]);
  document.getElementById('pf-count-lbl').textContent='Loading...';
  try{
    var selectedPts=getMs('ms-pf-pt');
    var typesToFetch=selectedPts.length?selectedPts:['Others','No Plate'];
    var allFetched=[];
    for(var i=0;i<typesToFetch.length;i++){
      var resp=await fetch('/api/plate_filter?plate_type='+encodeURIComponent(typesToFetch[i])+'&limit=500');
      var data=await resp.json();
      if(data.ok&&data.records.length) allFetched=allFetched.concat(data.records);
    }
    allFetched.sort(function(a,b){return b.id-a.id;});
    pfAllRows=allFetched;
    var filtered=applyTabFilter('pf',pfAllRows);
    renderPFTable(filtered);
    updateTabCount('pf',pfAllRows);
  }catch(e){
    document.getElementById('tbody-platefilter').innerHTML='<tr><td class="no-data" colspan="10">Error: '+e.message+'</td></tr>';
    document.getElementById('pf-count-lbl').textContent='Error';
  }
}
document.getElementById('btn-refresh-pf').addEventListener('click',fetchPlateFilter);

// ── Main Edit Modal (main table + Not Kenyan tab) ─────────────────────────────
var editModal=document.getElementById('edit-modal'),editInput=document.getElementById('edit-input');

document.getElementById('btn-edit').addEventListener('click',function(){
  if(selectedIdx===null||!tableRows[selectedIdx]){alert('Please select a row first.');return;}
  _inlineEditId=tableRows[selectedIdx].id;
  _inlineEditSource='main';
  editInput.value=tableRows[selectedIdx].modified||tableRows[selectedIdx].plate||'';
  editModal.classList.add('show');editInput.focus();editInput.select();
});

document.getElementById('edit-cancel').addEventListener('click',function(){
  editModal.classList.remove('show');_inlineEditId=null;_inlineEditSource=null;
});

document.getElementById('edit-save').addEventListener('click',async function(){
  var newText=editInput.value.trim().toUpperCase();
  if(!newText){editModal.classList.remove('show');return;}
  var src=_inlineEditSource, rid=_inlineEditId;
  editModal.classList.remove('show');_inlineEditId=null;_inlineEditSource=null;
  if(!rid){showToast('\u26a0 No record ID');return;}
  var newType=/^[A-Z]{3}\d{3}[A-Z]$/.test(newText)?'KE Plate':'Others';
  try{
    var resp=await fetch('/api/update_plate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:rid,text:newText})});
    if(!resp.ok)throw new Error('HTTP '+resp.status);
    var res=await resp.json();
    if(!res.ok){showToast('\u26a0 DB error: '+(res.error||'unknown'));return;}
    if(res.plate_type)newType=res.plate_type;
    showToast('\u2713 Saved: '+newText+' ('+newType+')');
  }catch(e){showToast('\u26a0 Save failed: '+e.message);return;}
  if(src==='main'&&selectedIdx!==null&&tableRows[selectedIdx]){
    var row=tableRows[selectedIdx];row.modified=newText;row.plateType=newType;row.isModified=true;row.isCorrect=false;
    renderTable();
  }
  var pfR=pfAllRows.find(function(r){return r.id===rid;});
  if(pfR){pfR.modified=newText;pfR.plateType=newType;pfR.was_modified=true;
    renderPFTable(applyTabFilter('pf',pfAllRows));updateTabCount('pf',pfAllRows);}
  [enteredRows,exitedRows].forEach(function(arr){
    var r=arr.find(function(r){return r.id===rid;});
    if(r){r.modified=newText;r.plateType=newType;r.isModified=true;}
  });
  renderTabTable('tbody-entered',applyTabFilter('entered',enteredRows));
  renderTabTable('tbody-exited',applyTabFilter('exited',exitedRows));
});

editModal.addEventListener('click',function(e){if(e.target===editModal){editModal.classList.remove('show');_inlineEditId=null;_inlineEditSource=null;}});
editInput.addEventListener('keydown',function(e){
  if(e.key==='Enter')document.getElementById('edit-save').click();
  if(e.key==='Escape'){editModal.classList.remove('show');_inlineEditId=null;_inlineEditSource=null;}
});

// ── Reports Edit Modal (separate, z-index 220, above reports overlay) ─────────
var _rptEditId=null;
var rptEditModal=document.getElementById('rpt-edit-modal');
var rptEditInput=document.getElementById('rpt-edit-input');

function openRptEdit(btn){
  _rptEditId=parseInt(btn.dataset.id);
  rptEditInput.value=(btn.dataset.plate||'').trim().toUpperCase();
  rptEditModal.classList.add('show');
  rptEditInput.focus();rptEditInput.select();
}

document.getElementById('rpt-edit-cancel').addEventListener('click',function(){
  rptEditModal.classList.remove('show');_rptEditId=null;
});

document.getElementById('rpt-edit-save').addEventListener('click',async function(){
  var newText=rptEditInput.value.trim().toUpperCase();
  if(!newText){rptEditModal.classList.remove('show');return;}
  var rid=_rptEditId;
  rptEditModal.classList.remove('show');_rptEditId=null;
  if(!rid){showToast('\u26a0 No record ID');return;}
  var newType=/^[A-Z]{3}\d{3}[A-Z]$/.test(newText)?'KE Plate':'Others';
  try{
    var resp=await fetch('/api/update_plate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:rid,text:newText})});
    if(!resp.ok)throw new Error('HTTP '+resp.status);
    var res=await resp.json();
    if(!res.ok){showToast('\u26a0 DB error: '+(res.error||'unknown'));return;}
    if(res.plate_type)newType=res.plate_type;
    showToast('\u2713 Saved: '+newText+' ('+newType+')');
  }catch(e){showToast('\u26a0 Save failed: '+e.message);return;}
  // Update local stores
  [tableRows,enteredRows,exitedRows].forEach(function(arr){
    var r=arr.find(function(r){return r.id===rid;});
    if(r){r.modified=newText;r.plateType=newType;r.isModified=true;}
  });
  var pfR=pfAllRows.find(function(r){return r.id===rid;});
  if(pfR){pfR.modified=newText;pfR.plateType=newType;pfR.was_modified=true;}
  // Refresh reports table
  if(rptLoaded)fetchReports();
});

rptEditModal.addEventListener('click',function(e){if(e.target===rptEditModal){rptEditModal.classList.remove('show');_rptEditId=null;}});
rptEditInput.addEventListener('keydown',function(e){
  if(e.key==='Enter')document.getElementById('rpt-edit-save').click();
  if(e.key==='Escape'){rptEditModal.classList.remove('show');_rptEditId=null;}
});

// ── Snapshot handling ─────────────────────────────────────────────────────────
async function onNewSnapshot(camIdx,deviceType,imageBlob){
  var b64=await new Promise(function(res,rej){
    var reader=new FileReader();
    reader.onload=function(){res(reader.result.split(',')[1]);};
    reader.onerror=rej;reader.readAsDataURL(imageBlob);
  });
  var saved=null;
  try{
    var resp=await fetch('/api/save_snapshot',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({cam_id:camIdx,device_type:deviceType,image_b64:b64})});
    saved=await resp.json();
    if(!saved.ok||saved.skipped)return;
  }catch(e){return;}
  var row={id:saved.id||null,date:eatDateStr(),time:eatTimeStr(),device:deviceType,
    dateFolder:saved.date_folder||null,largeName:saved.large_name||null,
    smallName:saved.small_name||null,
    cropName:(saved.crop_name&&saved.crop_name!=='')?saved.crop_name:null,
    conf:saved.plate_conf!=null?saved.plate_conf:null,
    plate:(saved.ocr_text||'').trim()||null,modified:null,
    plateType:saved.plate_type||'Others',isModified:false,isCorrect:true,
    enteredTime:saved.entered_time||null};
  tableRows.unshift(row);
  if(tableRows.length>MAX_LIVE)tableRows.pop();
  if(selectedIdx!==null)selectedIdx=Math.min(selectedIdx+1,tableRows.length-1);
  renderTable();
  var isEntry=deviceType.toLowerCase().indexOf('entry')>=0;
  var plateKey=(row.plate||'').trim().toUpperCase();
  if(isEntry){
    if(plateKey&&plateKey!=='K.'&&plateKey!=='K'){
      enteredRows=enteredRows.filter(function(e){return((e.plate||'').trim().toUpperCase())!==plateKey;});
    }
    enteredRows.unshift(row);
    if(enteredRows.length>MAX_TAB)enteredRows.pop();
    renderTabTable('tbody-entered',applyTabFilter('entered',enteredRows));
    updateTabCount('entered',enteredRows);
  }else{
    if(plateKey&&plateKey!=='K.'&&plateKey!=='K'){
      enteredRows=enteredRows.filter(function(e){return((e.plate||'').trim().toUpperCase())!==plateKey;});
      renderTabTable('tbody-entered',applyTabFilter('entered',enteredRows));
      updateTabCount('entered',enteredRows);
    }
    exitedRows.unshift(row);
    if(exitedRows.length>MAX_TAB)exitedRows.pop();
    renderTabTable('tbody-exited',applyTabFilter('exited',exitedRows));
    updateTabCount('exited',exitedRows);
  }
  fetchTodayCount();
}

var lastSnapSize=[0,0,0,0],lastSnapSaveMs=[0,0,0,0];
var SNAP_DEVICES=['Entry1','Entry2','Exit1','Exit2'];
var DEBOUNCE_MS=8000;

function pollSnapshot(idx){
  var url='/proxy_snap/'+idx+'?t='+Date.now();
  var card=document.getElementById('snap-card-'+idx);
  var body=document.getElementById('snap-body-'+idx);
  var dot=document.getElementById('snap-dot-'+idx);
  var tsEl=document.getElementById('snap-ts-'+idx);
  fetch(url).then(function(r){return r.blob();}).then(function(blob){
    var size=blob.size,objUrl=URL.createObjectURL(blob);
    body.innerHTML='';
    var img=document.createElement('img');img.src=objUrl;img.style.cursor='zoom-in';
    img.addEventListener('click',function(){openLightbox(objUrl,'Snapshot - '+SNAP_DEVICES[idx]);});
    body.appendChild(img);
    card.classList.remove('updated');void card.offsetWidth;card.classList.add('updated');
    dot.classList.add('active');tsEl.textContent=new Date().toTimeString().split(' ')[0];
    var changed=size>0&&size!==lastSnapSize[idx];
    var debounced=(Date.now()-lastSnapSaveMs[idx])>DEBOUNCE_MS;
    if(changed&&debounced){
      lastSnapSize[idx]=size;lastSnapSaveMs[idx]=Date.now();
      blob.arrayBuffer().then(function(buf){
        onNewSnapshot(idx,SNAP_DEVICES[idx],new Blob([buf],{type:'image/jpeg'}));
      });
    }
  }).catch(function(){dot.classList.remove('active');tsEl.textContent='No signal';});
}
function pollAll(){for(var i=0;i<4;i++)pollSnapshot(i);}
pollAll();setInterval(pollAll,3000);

// ── Reports ───────────────────────────────────────────────────────────────────
var rptOverlay=document.getElementById('rpt-overlay'),rptLimitEl=document.getElementById('rpt-limit'),rptAllBtn=document.getElementById('rpt-all-btn');
var rptState={from:'',to:'',search:'',filterMode:'',limitAll:false,rptTab:'all'};var rptLoaded=false;
function openReports(){rptOverlay.classList.add('show');if(rptLoaded)fetchReports();else triggerQuick('today');}
function closeReports(){rptOverlay.classList.remove('show');}
document.getElementById('btn-reports').addEventListener('click',openReports);
document.getElementById('rpt-close').addEventListener('click',closeReports);
document.getElementById('rpt-close-bottom').addEventListener('click',closeReports);
rptOverlay.addEventListener('click',function(e){if(e.target===rptOverlay)closeReports();});

document.querySelectorAll('.rpt-tab-btn').forEach(function(btn){
  btn.addEventListener('click',function(){
    document.querySelectorAll('.rpt-tab-btn').forEach(function(b){
      b.style.color='var(--muted)';b.style.borderBottom='3px solid transparent';
    });
    btn.style.color='var(--purple)';btn.style.borderBottom='3px solid var(--purple)';
    rptState.rptTab=btn.dataset.rpt;
    var isPF=btn.dataset.rpt==='platefilter';
    document.getElementById('pf-rpt-group').style.display=isPF?'flex':'none';
    if(rptLoaded)fetchReports();
  });
});

document.querySelectorAll('.qbtn[data-q]').forEach(function(btn){
  btn.addEventListener('click',function(){triggerQuick(btn.dataset.q);});
});
function triggerQuick(q){
  document.querySelectorAll('.qbtn').forEach(function(b){b.classList.remove('active');});
  var btn=document.querySelector('.qbtn[data-q="'+q+'"]');if(btn)btn.classList.add('active');
  var today=isoToday(),from,to;
  if(q==='today'){from=today;to=today;}
  else if(q==='yesterday'){from=isoOffset(-1);to=isoOffset(-1);}
  else if(q==='week'){var d=eatNow(),dow=d.getUTCDay(),diff=(dow===0)?-6:1-dow;from=isoOffset(diff);to=today;}
  else if(q==='month'){var d2=eatNow();from=d2.getUTCFullYear()+'-'+String(d2.getUTCMonth()+1).padStart(2,'0')+'-01';to=today;}
  else if(q==='last30'){from=isoOffset(-29);to=today;}
  document.getElementById('rpt-from').value=from;document.getElementById('rpt-to').value=to;
  rptState.from=from;rptState.to=to;fetchReports();
}
document.getElementById('fbtn-modified').addEventListener('click',function(){
  if(rptState.filterMode==='modified'){rptState.filterMode='';this.classList.remove('active');}
  else{rptState.filterMode='modified';this.classList.add('active');document.getElementById('fbtn-correct').classList.remove('active');}
  if(rptLoaded)fetchReports();
});
document.getElementById('fbtn-correct').addEventListener('click',function(){
  if(rptState.filterMode==='correct'){rptState.filterMode='';this.classList.remove('active');}
  else{rptState.filterMode='correct';this.classList.add('active');document.getElementById('fbtn-modified').classList.remove('active');}
  if(rptLoaded)fetchReports();
});
rptAllBtn.addEventListener('click',function(){
  rptState.limitAll=!rptState.limitAll;rptAllBtn.classList.toggle('active',rptState.limitAll);
  rptLimitEl.disabled=rptState.limitAll;rptLimitEl.style.opacity=rptState.limitAll?'0.4':'1';
  if(rptLoaded)fetchReports();
});
document.getElementById('rpt-generate').addEventListener('click',function(){
  rptState.from=document.getElementById('rpt-from').value;
  rptState.to=document.getElementById('rpt-to').value;
  rptState.search=document.getElementById('rpt-search').value.trim();
  document.querySelectorAll('.qbtn').forEach(function(b){b.classList.remove('active');});
  fetchReports();
});
document.getElementById('pf-rpt-select').addEventListener('change',function(){if(rptState.rptTab==='platefilter')fetchReports();});
document.getElementById('rpt-clear-search').addEventListener('click',function(){document.getElementById('rpt-search').value='';rptState.search='';if(rptLoaded)fetchReports();});
document.getElementById('rpt-search').addEventListener('keydown',function(e){if(e.key==='Enter')document.getElementById('rpt-generate').click();});
// Wire reports multi-select filters
['ms-rpt-device','ms-rpt-pt'].forEach(function(id){
  var wrap=document.getElementById(id);
  wrap.querySelectorAll('input[type=checkbox]').forEach(function(cb){
    cb.addEventListener('change',function(){updateMsLabel(id);if(rptLoaded)fetchReports();});
  });
});
document.getElementById('rpt-export-csv').addEventListener('click',function(){
  var f=document.getElementById('rpt-from').value,t=document.getElementById('rpt-to').value,s=document.getElementById('rpt-search').value.trim();
  var url='/api/export_csv?';if(f)url+='from='+f+'&';if(t)url+='to='+t+'&';if(s)url+='search='+encodeURIComponent(s);
  window.location.href=url;
});
document.getElementById('rpt-export-excel').addEventListener('click',function(){
  var f=document.getElementById('rpt-from').value,t=document.getElementById('rpt-to').value,s=document.getElementById('rpt-search').value.trim();
  var url='/api/export_csv?excel=1&';if(f)url+='from='+f+'&';if(t)url+='to='+t+'&';if(s)url+='search='+encodeURIComponent(s);
  window.location.href=url;
});
document.getElementById('rpt-export-pdf').addEventListener('click',function(){
  var f=document.getElementById('rpt-from').value,t=document.getElementById('rpt-to').value,s=document.getElementById('rpt-search').value.trim();
  var lp=rptState.limitAll?'all':(parseInt(rptLimitEl.value)>0?parseInt(rptLimitEl.value):100);
  var url='/api/export_pdf?limit='+lp;if(f)url+='&from='+f;if(t)url+='&to='+t;if(s)url+='&search='+encodeURIComponent(s);
  showToast('\u23f3 Generating PDF - please wait...');window.location.href=url;
});

// ── Reports sub-fetch helpers ─────────────────────────────────────────────────
async function _fetchRptInside(){
  document.getElementById('rpt-loading').style.display='flex';
  try{
    var resp=await fetch('/api/inside_cars'); var data=await resp.json();
    document.getElementById('rpt-loading').style.display='none';
    document.getElementById('rpt-stats').style.display='flex';
    document.getElementById('rpt-total').textContent=data.total||0;
    document.getElementById('rpt-unique').textContent=data.total||0;
    document.getElementById('rpt-range').textContent='Live - right now';
    document.getElementById('rpt-thead-tr').innerHTML='<th>Plate Number</th><th>Entry Date</th><th>Entry Time</th><th>Time Parked</th>';
    var tbody=document.getElementById('rpt-tbody');
    if(!data.ok||!data.records.length){tbody.innerHTML='<tr><td class="no-data" colspan="4">No vehicles currently inside.</td></tr>';}
    else{tbody.innerHTML=data.records.map(function(r){return'<tr><td><span class="ocr" style="font-size:14px;letter-spacing:2px">'+r.plate+'</span></td><td>'+formatDate(r.entry_date)+'</td><td><span style="color:#16a34a;font-weight:600">'+r.entry_time+'</span></td><td><span style="color:#7c3aed;font-weight:700">'+r.duration+'</span></td></tr>';}).join('');}
    document.getElementById('rpt-tbl-wrap').style.display='block';
    document.getElementById('rpt-showing').textContent='Showing '+(data.total||0)+' vehicles currently inside';
    rptLoaded=true;
  }catch(e){document.getElementById('rpt-loading').style.display='none';}
}

async function _fetchRptGhost(){
  document.getElementById('rpt-loading').style.display='flex';
  try{
    var resp=await fetch('/api/ghost_exits'); var data=await resp.json();
    document.getElementById('rpt-loading').style.display='none';
    document.getElementById('rpt-stats').style.display='flex';
    document.getElementById('rpt-total').textContent=data.total||0;
    document.getElementById('rpt-unique').textContent=data.total||0;
    document.getElementById('rpt-range').textContent='All time';
    document.getElementById('rpt-thead-tr').innerHTML='<th>Plate Number</th><th>Exit Date</th><th>Exit Time</th>';
    var tbody=document.getElementById('rpt-tbody');
    if(!data.ok||!data.records.length){tbody.innerHTML='<tr><td class="no-data" colspan="3">No unregistered exits found.</td></tr>';}
    else{tbody.innerHTML=data.records.map(function(r){return'<tr><td><span class="ocr" style="font-size:14px;letter-spacing:2px">'+r.plate+'</span></td><td>'+formatDate(r.exit_date)+'</td><td><span style="color:#ea580c;font-weight:600">'+r.exit_time+'</span></td></tr>';}).join('');}
    document.getElementById('rpt-tbl-wrap').style.display='block';
    document.getElementById('rpt-showing').textContent='Showing '+(data.total||0)+' unregistered exits';
    rptLoaded=true;
  }catch(e){document.getElementById('rpt-loading').style.display='none';}
}

async function _fetchRptPlateFilter(){
  var pt=document.getElementById('pf-rpt-select').value;
  var from=document.getElementById('rpt-from').value;
  var to=document.getElementById('rpt-to').value;
  var lp=rptState.limitAll?'all':(parseInt(document.getElementById('rpt-limit').value)>0?parseInt(document.getElementById('rpt-limit').value):100);
  document.getElementById('rpt-loading').style.display='flex';
  try{
    var url='/api/plate_filter?plate_type='+encodeURIComponent(pt)+'&limit='+lp;
    if(from)url+='&from='+from; if(to)url+='&to='+to;
    var resp=await fetch(url); var data=await resp.json();
    document.getElementById('rpt-loading').style.display='none';
    document.getElementById('rpt-stats').style.display='flex';
    document.getElementById('rpt-total').textContent=data.total||0;
    document.getElementById('rpt-unique').textContent=data.total||0;
    document.getElementById('rpt-range').textContent=pt;
    document.getElementById('rpt-thead-tr').innerHTML='<th>ID</th><th>Date</th><th>Time</th><th>Device</th><th>Vehicle</th><th>Plate</th><th>OCR Text</th><th>Modified Text</th><th>Type</th><th>Conf%</th><th>Edit</th>';
    var tbody=document.getElementById('rpt-tbody');
    if(!data.ok||!data.records.length){tbody.innerHTML='<tr><td class="no-data" colspan="11">No records for '+pt+'.</td></tr>';}
    else{
      tbody.innerHTML=data.records.map(function(r){
        var ls=(r.dateFolder&&r.largeName)?'/snapshots/'+r.dateFolder+'/large/'+r.largeName:'';
        var cs=(r.cropName&&r.dateFolder)?'/snapshots/'+r.dateFolder+'/crop/'+r.cropName:'';
        var lT=ls?'<div class="rpt-thumb"><img src="'+ls+'" data-src="'+ls+'" data-lbl="'+r.device+'" onerror="this.parentElement.textContent=\'-\'"></div>':'<div class="rpt-thumb" style="font-size:10px;color:#9ca3af">-</div>';
        var pT=cs?'<div class="rpt-thumb-crop"><img src="'+cs+'" data-src="'+cs+'" data-lbl="'+(r.plate||'')+'" onerror="this.parentElement.textContent=\'-\'"></div>':'<div class="rpt-thumb-crop" style="font-size:10px;color:#9ca3af">-</div>';
        var isExit=(r.device||'').toLowerCase().indexOf('exit')>=0;
        var currentPlate=(r.modified&&r.was_modified)?r.modified:r.plate;
        var ocrCell=r.plate?'<span class="ocr">'+r.plate+'</span>':'<span style="color:#9ca3af">-</span>';
        var editBtn=r.id?'<button class="inline-edit-btn" data-id="'+r.id+'" data-plate="'+(currentPlate||'')+'" onclick="openRptEdit(this)">&#9999; Edit</button>':'<span style="color:#9ca3af">-</span>';
        return'<tr><td><b>'+r.id+'</b></td><td>'+formatDate(r.date)+'</td><td>'+r.time+'</td>'+
          '<td><span class="badge'+(isExit?' exit':'')+'">'+r.device+'</span></td>'+
          '<td style="padding:3px 6px">'+lT+'</td><td style="padding:3px 6px">'+pT+'</td>'+
          '<td>'+ocrCell+'</td>'+
          '<td>'+(r.modified&&r.was_modified?'<span class="ocr-edit">'+r.modified+'</span>':'<span style="color:#9ca3af">-</span>')+'</td>'+
          '<td>'+plateBadge(r.plateType)+'</td><td>'+(r.conf!=null?r.conf+'%':'-')+'</td>'+
          '<td style="text-align:center">'+editBtn+'</td></tr>';
      }).join('');
      attachRptImageListeners();
    }
    document.getElementById('rpt-tbl-wrap').style.display='block';
    document.getElementById('rpt-showing').textContent='Showing '+(data.total||0)+' records for '+pt;
    rptLoaded=true;
  }catch(e){document.getElementById('rpt-loading').style.display='none';}
}

function attachRptImageListeners(){
  document.querySelectorAll('#rpt-tbody img[data-src]').forEach(function(img){
    if(img._lbAttached)return; img._lbAttached=true;
    img.addEventListener('click',function(e){e.stopPropagation();openLightbox(img.dataset.src,img.dataset.lbl);});
  });
}

// ── Reports OCR retry in table ────────────────────────────────────────────────
function rptOcrRetryBtn(rid, hasCrop){
  if(!hasCrop||!rid)return'';
  return '<button class="ocr-retry-btn" title="Retry OCR" data-id="'+rid+'" onclick="retryOcr(this)">'+RETRY_SVG+'</button>';
}

async function fetchReports(){
  var from=document.getElementById('rpt-from').value||rptState.from;
  var to=document.getElementById('rpt-to').value||rptState.to;
  var search=document.getElementById('rpt-search').value.trim()||rptState.search;
  var lp=rptState.limitAll?'all':(parseInt(rptLimitEl.value)>0?parseInt(rptLimitEl.value):100);
  // Collect multi-select filters for reports
  var rptDevices=getMs('ms-rpt-device');
  var rptPts=getMs('ms-rpt-pt');
  rptState.from=from;rptState.to=to;rptState.search=search;
  ['rpt-welcome','rpt-stats','rpt-tbl-wrap','rpt-empty'].forEach(function(id){document.getElementById(id).style.display='none';});
  document.getElementById('rpt-loading').style.display='flex';
  document.getElementById('rpt-showing').textContent='';
  if(rptState.rptTab==='inside'){document.getElementById('rpt-loading').style.display='none';_fetchRptInside();return;}
  if(rptState.rptTab==='ghost'){document.getElementById('rpt-loading').style.display='none';_fetchRptGhost();return;}
  if(rptState.rptTab==='platefilter'){document.getElementById('rpt-loading').style.display='none';_fetchRptPlateFilter();return;}
  var url='/api/reports?limit='+lp;
  if(from)url+='&from='+from;if(to)url+='&to='+to;
  if(search)url+='&search='+encodeURIComponent(search);
  if(rptState.filterMode)url+='&filter='+rptState.filterMode;
  // Device: multi
  if(rptDevices.length)url+='&device='+encodeURIComponent(rptDevices.join(','));
  else if(rptState.rptTab==='entered')url+='&device='+encodeURIComponent('Entry1,Entry2');
  else if(rptState.rptTab==='exited')url+='&device='+encodeURIComponent('Exit1,Exit2');
  // Plate type: multi
  if(rptPts.length)url+='&plate_type='+encodeURIComponent(rptPts.join(','));
  var data=null;
  try{var resp=await fetch(url);data=await resp.json();}
  catch(err){
    document.getElementById('rpt-loading').style.display='none';
    document.getElementById('rpt-empty').innerHTML='<div style="color:#dc2626;font-weight:700">Network error: '+err.message+'</div>';
    document.getElementById('rpt-empty').style.display='flex';return;
  }
  document.getElementById('rpt-loading').style.display='none';
  rptLoaded=true;
  if(!data||!data.ok){
    document.getElementById('rpt-empty').innerHTML='<div style="color:#dc2626;font-weight:700">Error: '+(data&&data.error?data.error:'Unknown')+'</div>';
    document.getElementById('rpt-empty').style.display='flex';return;
  }
  document.getElementById('rpt-total').textContent=data.total;
  document.getElementById('rpt-unique').textContent=data.unique;
  document.getElementById('rpt-range').textContent=(data.date_from&&data.date_to)?formatDate(data.date_from)+' - '+formatDate(data.date_to):'All dates';
  document.getElementById('rpt-stats').style.display='flex';
  if(!data.records||!data.records.length){
    document.getElementById('rpt-empty').textContent='No records found.';
    document.getElementById('rpt-empty').style.display='flex';
    document.getElementById('rpt-showing').textContent='No records found';return;
  }
  document.getElementById('rpt-thead-tr').innerHTML='<th>ID</th><th>Date</th><th>Time</th><th>Device</th><th>Vehicle</th><th>Plate</th><th>OCR Text</th><th>Modified Text</th><th>Modified?</th><th>Correct?</th><th>Type</th><th>Conf%</th><th>Edit</th>';
  var tbody=document.getElementById('rpt-tbody');
  tbody.innerHTML=data.records.map(function(r){
    var isExit=(r.device||'').toLowerCase().indexOf('exit')>=0;
    var devBadge='<span class="badge'+(isExit?' exit':'')+'">'+r.device+'</span>';
    var lThumb=r.large_url?'<div class="rpt-thumb"><img src="'+r.large_url+'" data-src="'+r.large_url+'" data-lbl="Vehicle '+r.device+'" onerror="this.parentElement.textContent=\'-\'"></div>':'<div class="rpt-thumb" style="font-size:10px;color:#9ca3af">-</div>';
    var hasCrop=!!r.crop_url;
    var cThumb=hasCrop?'<div class="rpt-thumb-crop"><img src="'+r.crop_url+'" data-src="'+r.crop_url+'" data-lbl="Plate '+(r.display_plate||'')+'" onerror="this.parentElement.textContent=\'-\'"></div>':'<div class="rpt-thumb-crop" style="font-size:10px;color:#9ca3af">-</div>';
    var ocrTd=(r.plate?'<span class="ocr">'+r.plate+'</span>':'<span style="color:#9ca3af;font-style:italic;font-size:11px">No OCR</span>')+rptOcrRetryBtn(r.id,hasCrop);
    var modTd=r.modified?'<span class="ocr-edit">'+r.modified+'</span>':'<span style="color:#d1d5db">\u2014</span>';
    var modInd=r.was_modified?'<span class="mod-yes">\u2713</span>':'<span class="mod-no">\u2014</span>';
    var chkCls=r.is_correct?' checked':'';
    var correctBtn='<button class="correct-btn'+chkCls+'" data-id="'+r.id+'" data-correct="'+(r.is_correct?'1':'0')+'">\u2713</button>';
    var confTd=r.conf!=null?r.conf+'%':'\u2014';
    var currentPlate=(r.modified&&r.was_modified)?r.modified:r.plate;
    var editBtn=r.id?'<button class="inline-edit-btn" data-id="'+r.id+'" data-plate="'+(currentPlate||'')+'" onclick="openRptEdit(this)">&#9999; Edit</button>':'\u2014';
    return'<tr><td><b>'+r.id+'</b></td><td style="white-space:nowrap">'+formatDate(r.date)+'</td><td style="white-space:nowrap">'+r.time+'</td><td>'+devBadge+'</td><td style="padding:3px 6px">'+lThumb+'</td><td style="padding:3px 6px">'+cThumb+'</td><td>'+ocrTd+'</td><td>'+modTd+'</td><td style="text-align:center">'+modInd+'</td><td style="text-align:center">'+correctBtn+'</td><td>'+plateBadge(r.plate_type)+'</td><td>'+confTd+'</td><td style="text-align:center">'+editBtn+'</td></tr>';
  }).join('');
  attachRptImageListeners();
  document.querySelectorAll('#rpt-tbody .correct-btn').forEach(function(btn){
    btn.addEventListener('click',async function(e){
      e.stopPropagation();
      var rid=parseInt(btn.dataset.id),newOk=btn.dataset.correct!=='1';
      btn.dataset.correct=newOk?'1':'0';btn.classList.toggle('checked',newOk);
      try{
        await fetch('/api/mark_correct',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:rid,is_correct:newOk})});
        showToast(newOk?'\u2713 Marked correct':'\u2717 Unmarked');
      }catch(err){showToast('\u26a0 Update failed');}
    });
  });
  document.getElementById('rpt-tbl-wrap').style.display='block';
  document.getElementById('rpt-showing').textContent='Showing '+data.records.length+' of '+data.total+' records';
}

// ── Draggable divider ─────────────────────────────────────────────────────────
(function(){
  var handle=document.getElementById('drag-handle');
  var panel=handle.closest('.panel-table');
  var latest=panel.querySelector('.latest-section');
  var dragging=false,startY=0,startH=0,totalH=0;
  handle.addEventListener('mousedown',function(e){
    dragging=true;startY=e.clientY;
    startH=latest.getBoundingClientRect().height;
    totalH=panel.getBoundingClientRect().height;
    document.body.style.cursor='row-resize';
    document.body.style.userSelect='none';
    e.preventDefault();
  });
  document.addEventListener('mousemove',function(e){
    if(!dragging)return;
    var dy=e.clientY-startY;
    var newH=Math.max(80,Math.min(startH+dy,totalH-80));
    latest.style.flex='0 0 '+((newH/totalH)*100)+'%';
  });
  document.addEventListener('mouseup',function(){
    if(!dragging)return;
    dragging=false;document.body.style.cursor='';document.body.style.userSelect='';
  });
})();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(_HTML_CONTENT)

@app.route("/video/<int:cam_id>")
def video_feed(cam_id):
    if cam_id not in frames:
        return "Camera not found", 404
    return Response(generate(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("\n AMAAN ANPR Dashboard")
    print(" ---------------------")
    if not os.path.exists(ONNX_PATH):
        print(f" !! WARNING: best.onnx not found at {ONNX_PATH}")
    else:
        print(f" -> best.onnx found ({os.path.getsize(ONNX_PATH)//1024} KB)")
    print(f" Snapshots: {SNAPSHOT_BASE}")
    for cam in CAMERAS:
        t = threading.Thread(target=capture_frames, args=(cam,), daemon=True)
        t.start()
    load_onnx()
    load_ocr()
    try:
        conn = get_db(); conn.close()
        print(" -> Database connected OK")
        _ensure_all_cols()
    except Exception as e:
        print(f" !! Database FAILED: {e}")
    print("\n Open: http://localhost:8888\n")
    app.run(host="0.0.0.0", port=8888, debug=False, threaded=True)