import cv2
import json
import sys
from pathlib import Path

# ------------------------------
# ตั้งค่าเริ่มต้น
# ------------------------------
IMAGE_PATH = r"D:\WORKFing\NLP\minipro\slip_images\1759752941870.jpg"
OUTPUT_JSON = "roi_output.json"          # ไฟล์ผลลัพธ์ JSON
TEMPLATE_BANK_CODE = "KTB"               # ปรับได้
TESS_LANG = "tha+eng"                    # ปรับได้
DEFAULT_PSM = 7                          # ปรับได้ (7 = single line/word)
SHOW_WIDTH = 500                        # ความกว้างที่แสดง (ย่อเฉพาะหน้าจอ)

# ------------------------------
# โหลดภาพ + เตรียมสเกลแสดงผล
# ------------------------------
img = cv2.imdecode(
    np.fromfile(IMAGE_PATH, dtype=np.uint8), cv2.IMREAD_COLOR
) if 'np' in globals() else cv2.imread(IMAGE_PATH)

if img is None:
    raise SystemExit(f"ไม่พบรูป: {IMAGE_PATH}")

import numpy as np
H, W = img.shape[:2]
scale = SHOW_WIDTH / W if W > SHOW_WIDTH else 1.0
disp = cv2.resize(img, (int(W*scale), int(H*scale))) if scale != 1.0 else img.copy()

# ------------------------------
# เก็บกรอบและชื่อฟิลด์
# ------------------------------
boxes = []  # [(x1,y1,x2,y2), ...] บนพิกัด "ต้นฉบับ"
fields = [] # ชื่อฟิลด์ตามกล่อง

pt1 = None
pt2 = None
drawing = False

def draw_overlay():
    overlay = disp.copy()
    # วาดทุกกรอบที่บันทึกแล้ว
    for i,(x1,y1,x2,y2) in enumerate(boxes):
        a = int(x1*scale); b = int(y1*scale); c = int(x2*scale); d = int(y2*scale)
        cv2.rectangle(overlay, (a,b), (c,d), (0,255,0), 2)
        tag = fields[i] if i < len(fields) and fields[i] else f"roi_{i+1}"
        cv2.putText(overlay, tag, (a, max(20,b-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    # วาดกรอบกำลังเลือก
    if pt1 and drawing:
        cv2.rectangle(overlay, pt1, current_mouse, (255,200,0), 2)
    cv2.imshow("ROI Picker", overlay)

current_mouse = (0,0)

def on_mouse(event, x, y, flags, param):
    global pt1, pt2, drawing, current_mouse
    current_mouse = (x,y)
    if event == cv2.EVENT_LBUTTONDOWN:
        pt1 = (x,y)
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        pt2 = (x,y)
        drawing = False

cv2.namedWindow("ROI Picker")
cv2.setMouseCallback("ROI Picker", on_mouse)

print("\n=== ROI Picker ===")
print(f"ภาพ: {IMAGE_PATH} (ต้นฉบับ {W}x{H} px, แสดงผลสเกล {scale:.3f})")
print("วิธีใช้: คลิกซ้าย 2 ครั้ง (ซ้ายบน → ขวาล่าง) แล้วกด ENTER เพื่อยืนยันกรอบ")
print("คีย์: ENTER=ยืนยันกรอบ  |  Z=ย้อน 1 กรอบ  |  C=ล้างทั้งหมด  |  S=เซฟ JSON  |  Q=ออก\n")

while True:
    draw_overlay()
    key = cv2.waitKey(10) & 0xFF

    if key == 13 or key == 10:  # ENTER
        if pt1 and pt2:
            # จัดตำแหน่งให้ pt1 = มุมซ้ายบน, pt2 = ขวาล่าง
            (x1, y1) = pt1
            (x2, y2) = pt2
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            # แปลงกลับเป็นพิกัด "ต้นฉบับ"
            X1 = int(x1 / scale); Y1 = int(y1 / scale)
            X2 = int(x2 / scale); Y2 = int(y2 / scale)

            # คำนวณสัมพัทธ์
            x_rel = X1 / W
            y_rel = Y1 / H
            w_rel = (X2 - X1) / W
            h_rel = (Y2 - Y1) / H

            print("\n--- ROI ใหม่ ---")
            print(f"px : X={X1}, Y={Y1}, W={X2-X1}, H={Y2-Y1}")
            print(f"rel: [ {x_rel:.4f}, {y_rel:.4f}, {w_rel:.4f}, {h_rel:.4f} ]")

            # ขอชื่อฟิลด์ไว้ทำ template เช่น amount, datetime, payer
            field_name = input("ตั้งชื่อฟิลด์ (เช่น amount / datetime / payer) เว้นว่างเพื่อข้าม: ").strip()
            boxes.append((X1,Y1,X2,Y2))
            fields.append(field_name if field_name else f"roi_{len(boxes)}")

            pt1, pt2 = None, None

    elif key in (ord('z'), ord('Z')):  # undo
        if boxes:
            boxes.pop()
            if fields: fields.pop()
            print("ย้อน 1 กรอบแล้ว")
    elif key in (ord('c'), ord('C')):  # clear
        boxes.clear(); fields.clear()
        print("ล้างกรอบทั้งหมดแล้ว")
    elif key in (ord('s'), ord('S')):  # save JSON
        # แปลงเป็น JSON เทมเพลต
        rois = {}
        for i,(x1,y1,x2,y2) in enumerate(boxes):
            x_rel = x1/W; y_rel = y1/H; w_rel = (x2-x1)/W; h_rel = (y2-y1)/H
            rois[fields[i]] = {"box":[round(x_rel,4), round(y_rel,4), round(w_rel,4), round(h_rel,4)],
                               "psm": DEFAULT_PSM}
        template = {
            "bank_code": TEMPLATE_BANK_CODE,
            "tess_lang": TESS_LANG,
            "rois": rois,
            "preprocess": { "upscale": 1.5 }  # แก้ไขได้ตามต้องการ
        }
        Path(OUTPUT_JSON).write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"บันทึก JSON → {OUTPUT_JSON}")
    elif key in (ord('q'), ord('Q'), 27):  # ESC/Q quit
        break

cv2.destroyAllWindows()
