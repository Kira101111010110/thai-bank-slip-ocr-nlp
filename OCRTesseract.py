# Balanced OCR with selective noise reduction

import cv2, json, re, numpy as np, pytesseract
from pathlib import Path
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================================
#  Helper Functions
# =========================================

def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def normalize_thai_spacing(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r'(?P<t1>[ก-๙])\s+(?=[ก-๙])', r'\g<t1>', s)
    for _ in range(3):
        s = re.sub(r'([ก-๙])\s+(?=[ก-๙])', r'\1', s)
    return re.sub(r'[ \t]+', ' ', s).strip()


def multi_variant_preprocess(gray, scale=3.0):
    """Generate multiple preprocessing variants, let OCR scoring pick best"""
    variants = []
    
    # Upscale
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Variant 1: Simple CLAHE + Adaptive Threshold
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blur1 = cv2.GaussianBlur(enhanced, (3,3), 0)
    binary1 = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    variants.append(('adaptive_gauss', binary1))
    
    # Variant 2: Otsu threshold (good for clear text)
    blur2 = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(('otsu', binary2))
    
    # Variant 3: Mean adaptive (different from Gaussian)
    blur3 = cv2.GaussianBlur(enhanced, (3,3), 0)
    binary3 = cv2.adaptiveThreshold(blur3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 15, 5)
    variants.append(('adaptive_mean', binary3))
    
    # Variant 4: Bilateral filter + adaptive (preserves edges)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    bi_enhanced = clahe2.apply(bilateral)
    blur4 = cv2.GaussianBlur(bi_enhanced, (3,3), 0)
    binary4 = cv2.adaptiveThreshold(blur4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 13, 3)
    variants.append(('bilateral', binary4))
    
    # Variant 5: Simple threshold with different cutoff
    _, binary5 = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
    variants.append(('simple_thresh', binary5))
    
    return variants


def crop_rel(image, box_rel):
    H, W = image.shape[:2]
    x, y, w, h = box_rel
    x1, y1 = max(0,int(x*W)), max(0,int(y*H))
    x2, y2 = min(W,int((x+w)*W)), min(H,int((y+h)*H))
    return image[y1:y2, x1:x2]


def clean_ocr_text(s: str) -> str:
    if s is None:
        return ""
    s = re.sub(r'[\x00-\x1f\x7f]+', ' ', s)
    s = s.replace('\r','\n')
    s = re.sub(r'\n+', '\n', s)
    s = s.strip(' \t\n\r\ufeff\u200b')
    return s


def normalize_one_line(s: str) -> str:
    s = clean_ocr_text(s)
    return ' '.join([ln.strip() for ln in s.splitlines() if ln.strip()])


def tesseract_read_single(img, lang="tha+eng", psm=7, whitelist=None):
    cfg = f"--oem 1 --psm {psm} --dpi 300"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    try:
        txt = pytesseract.image_to_string(img, lang=lang, config=cfg)
    except Exception as e:
        txt = ''
    return clean_ocr_text(txt)


def tesseract_read_variants(variants, lang="tha+eng", psms=(6,7), whitelist=None, field=None):
    """Try multiple preprocessing variants AND PSM modes, score all combinations"""
    candidates = []
    
    for variant_name, img in variants:
        # Try normal and inverted
        for invert_name, test_img in [('normal', img), ('inverted', 255 - img)]:
            for psm in psms:
                txt = tesseract_read_single(test_img, lang=lang, psm=psm, whitelist=whitelist)
                candidates.append({
                    'txt': txt,
                    'variant': f"{variant_name}_{invert_name}",
                    'psm': psm
                })
    
    # Score candidates
    def score_candidate(txt: str) -> float:
        if not txt:
            return -1000.0
        one = normalize_one_line(txt)
        if not one:
            return -1000.0
        
        if field in ('amount','fee'):
            # Look for numbers with decimals
            digits = len(re.findall(r'\d', one))
            if digits == 0:
                return -500.0
            
            decimal_match = re.search(r'\d+\.\d{2}', one)
            if decimal_match:
                return 1000.0 + digits * 10
            
            # Numbers without decimal
            has_nums = bool(re.search(r'\d+', one))
            noise = len(re.findall(r'[^0-9\. ]', one))
            return (digits * 20) - (noise * 10) if has_nums else -200.0
            
        elif field in ('payer','payee'):
            # Must have Thai characters
            thai_chars = len(re.findall(r'[ก-๙]', one))
            if thai_chars < 3:
                return -300.0
            
            spaces = one.count(' ')
            total_len = len(one)
            # Prefer 2-3 word names
            word_bonus = 100 if 1 <= spaces <= 3 else 0
            return thai_chars * 15 + word_bonus + total_len
            
        elif field == 'datetime':
            parsed = parse_datetime_th(one)
            if parsed:
                return 2000.0
            # Partial scoring
            has_date = bool(re.search(r'\d{1,2}', one))
            has_time = bool(re.search(r'\d{1,2}[:\.]\d{2}', one))
            has_year = bool(re.search(r'25\d{2}', one))
            return (has_date * 50) + (has_time * 50) + (has_year * 100)
            
        elif field == 'reference':
            # Look for long alphanumeric
            long_match = re.search(r'[A-Za-z0-9]{12,}', one)
            if long_match:
                return 1500.0
            med_match = re.search(r'[A-Za-z0-9]{8,}', one)
            if med_match:
                return 800.0
            return len(one) if len(one) > 5 else -100.0
        
        else:
            # Generic: length matters
            return len(one) * 2
    
    best = max(candidates, key=lambda c: score_candidate(c['txt']))
    result = normalize_one_line(best['txt'])
    
    # Debug: print top 3 candidates
    sorted_cands = sorted(candidates, key=lambda c: score_candidate(c['txt']), reverse=True)[:3]
    
    return result, sorted_cands


def extract_number(text):
    """Extract monetary amount with decimal support"""
    if not text:
        return 0.0
    
    t = text.replace(',', '').replace('บาท','').replace('B','').strip()
    
    # Look for decimal numbers first
    decimal_match = re.search(r'(\d+)\.(\d{2})', t)
    if decimal_match:
        return float(decimal_match.group(0))
    
    # Look for any number
    nums = re.findall(r'\d+', t)
    if not nums:
        return 0.0
    
    try:
        # Single number
        if len(nums) == 1:
            return float(nums[0])
        
        # If we have exactly 2 numbers and second is 2 digits, treat as decimal
        if len(nums) == 2 and len(nums[1]) == 2:
            return float(f"{nums[0]}.{nums[1]}")
        
        # Otherwise concatenate (e.g., "1 2 3 4" -> 1234)
        return float(''.join(nums))
    except:
        return 0.0


def parse_datetime_th(s: str):
    if not s:
        return None
    s = normalize_one_line(s)
    
    TH_MONTHS = {
        "ม.ค.":1,"ก.พ.":2,"มี.ค.":3,"เม.ย.":4,"พ.ค.":5,"มิ.ย.":6,
        "ก.ค.":7,"ส.ค.":8,"ก.ย.":9,"ต.ค.":10,"พ.ย.":11,"ธ.ค.":12,
        "ม.ค":1,"ก.พ":2,"มี.ค":3,"เม.ย":4,"พ.ค":5,"มิ.ย":6,
        "ก.ค":7,"ส.ค":8,"ก.ย":9,"ต.ค":10,"พ.ย":11,"ธ.ค":12
    }

    patterns = [
        r'(\d{1,2})\s*([ก-๙.]+)\s*(\d{4})[\s,\-]*?(\d{1,2})[:.\-](\d{2})',
        r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})[\s,\-]*(\d{1,2})[:.\-](\d{2})',
        r'(\d{1,2})\s*([ก-๙.]+)\s*(\d{4})',
    ]

    for pat in patterns:
        m = re.search(pat, s)
        if m:
            groups = m.groups()
            try:
                if len(groups) >= 5:
                    d, mon_txt, yy, hh, mi = groups
                    d, yy, hh, mi = int(d), int(yy), int(hh), int(mi)
                    mm = TH_MONTHS.get(mon_txt.strip())
                    if mm and yy > 2400:
                        yy -= 543
                    if mm:
                        return datetime(yy, mm, d, hh, mi).isoformat()
                elif len(groups) == 3:
                    d, mon_txt, yy = groups
                    d, yy = int(d), int(yy)
                    mm = TH_MONTHS.get(mon_txt.strip())
                    if mm and yy > 2400:
                        yy -= 543
                    if mm:
                        return datetime(yy, mm, d).isoformat()
            except Exception:
                continue
    
    return None


# =========================================
#  OCR by Template
# =========================================

def run_with_template(image_path, tpl):
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lang = tpl.get("tess_lang", "tha+eng")

    default_whitelists = {
        'amount': '0123456789.',
        'fee': '0123456789.',
        'datetime': '0123456789ก-๙./: -',
        'payer': '',
        'payee': '',
        'reference': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    }

    raw = {}
    debug_dir = Path('debug_crops')
    debug_dir.mkdir(exist_ok=True)
    
    for i, (field, meta) in enumerate(tpl["rois"].items(), start=1):
        roi_gray = crop_rel(gray, meta["box"])
        
        # Generate multiple preprocessing variants
        variants = multi_variant_preprocess(roi_gray, scale=3.0)
        
        # Save the first variant for visual debugging
        cv2.imencode('.png', variants[0][1])[1].tofile(str(debug_dir / f"{i:02d}_{field}.png"))

        whitelist = meta.get('whitelist', default_whitelists.get(field))
        
        # Field-specific PSM strategies
        if field in ('amount', 'fee'):
            psms_to_try = [7, 8, 13]
        elif field in ('payer', 'payee'):
            psms_to_try = [6, 7, 3]
        elif field == 'reference':
            psms_to_try = [7, 8, 13]
        elif field == 'datetime':
            psms_to_try = [6, 7]
        else:
            psms_to_try = [6, 7]

        txt, top_candidates = tesseract_read_variants(
            variants, lang=lang, psms=psms_to_try,
            whitelist=whitelist if whitelist else None,
            field=field
        )

        # Save debug info
        debug_text = f"Best result: {txt}\n\nTop 3 candidates:\n"
        for j, cand in enumerate(top_candidates, 1):
            debug_text += f"{j}. [{cand['variant']}, PSM{cand['psm']}]: {cand['txt']}\n"
        
        (debug_dir / f"{i:02d}_{field}.txt").write_text(debug_text, encoding='utf-8')
        raw[field] = txt

    # Parse results
    ref_txt = raw.get('reference', '')
    ref_m = re.search(r'[A-Za-z0-9]{10,}', ref_txt)
    if not ref_m:
        ref_m = re.search(r'[A-Za-z0-9]{8,}', ref_txt)
    reference = ref_m.group(0) if ref_m else None

    out = {
        "payment": tpl.get("bank_code"),
        "reference": reference,
        "payer": raw.get("payer") or None,
        "payee": raw.get("payee") or None,
        "money": extract_number(raw.get("amount", "")),
        "fee": extract_number(raw.get("fee", "")),
        "time": parse_datetime_th(raw.get("datetime", ""))
    }

    return out, raw


# =========================================
#  Show ROI Preview
# =========================================

def visualize_rois(image_path, tpl, show_width=500):
    img = load_image(image_path)
    for name, meta in tpl["rois"].items():
        x, y, w, h = meta["box"]
        H, W = img.shape[:2]
        x1, y1 = int(x*W), int(y*H)
        x2, y2 = int((x+w)*W), int((y+h)*H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    scale = show_width / img.shape[1]
    resized = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    cv2.imshow("Template ROI Preview", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================================
#  MAIN
# =========================================
if __name__ == "__main__":
    template_path = Path("templates/KTB.json")
    image_path = Path("slip_images/1759752941870.jpg")

    tpl = json.loads(template_path.read_text(encoding="utf-8"))

    # Show ROI boxes
    visualize_rois(image_path, tpl, show_width=500)

    # Run OCR
    clean_json, raw_fields = run_with_template(image_path, tpl)

    print("\n=== RAW FIELDS ===")
    for k,v in raw_fields.items():
        print(f"{k}: {v}")
    print("\n=== CLEAN JSON ===")
    print(json.dumps(clean_json, ensure_ascii=False, indent=2))