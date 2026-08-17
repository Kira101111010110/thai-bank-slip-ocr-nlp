"""
Slip OCR & NLP (Universal) - Improved Version
- Accepts: file path, bytes, numpy array, PIL.Image
- Optional: HEIC/HEIF (via pillow-heif), PDF (first page via pdf2image)
- Robust preprocessing: EXIF autorotate, deskew (fast), adaptive threshold, denoise, multi-rotation trials
- OCR: EasyOCR (th + en)
- NLP: pythainlp (word_tokenize) for tokenizing; regex for fields
"""

from __future__ import annotations
import io
import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Literal
from datetime import datetime

import numpy as np
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional imports guarded
try:
    from PIL import Image, ImageOps, ExifTags
except Exception:
    Image = None
    ImageOps = None
    ExifTags = None

try:
    import pillow_heif  # for HEIC/HEIF support
    pillow_heif.register_heif_opener()
except Exception:
    pass

# Optional: PDF to images (first page)
def _pdf_to_image_first_page(pdf_path: str) -> Optional["Image.Image"]:
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
        if pages:
            return pages[0]
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        return None
    return None

# OCR & Thai NLP
import easyocr
from pythainlp.tokenize import word_tokenize
from pythainlp import Tokenizer


ImgLike = Union[str, bytes, np.ndarray, "Image.Image"]


class SlipConfig:
    """Configuration constants for slip processing"""
    BANK_KEYWORDS = ['โอนเงิน', 'ผู้โอน', 'ผู้รับ', 'จำนวนเงิน', 'ธนาคาร',
                     'บัญชี', 'เลขที่อ้างอิง', 'วันที่', 'เวลา', 'ยอดเงิน',
                     'ค่าธรรมเนียม', 'คงเหลือ', 'พร้อมเพย์', 'อ้างอิง', 'Reference']
    
    BANK_NAMES = {
        'กรุงเทพ': 'BBL',
        'กสิกรไทย': 'KBANK',
        'ไทยพาณิชย์': 'SCB',
        'กรุงไทย': 'KTB',
        'ทหารไทย': 'TTB',
        'กรุงศรี': 'BAY',
        'ออมสิน': 'GSB',
        'ธกส': 'BAAC',
        'ธนชาต': 'TBANK',
        'ซีไอเอ็มบี': 'CIMB',
        'ยูโอบี': 'UOB',
        'แลนด์': 'LHBANK'
    }
    
    OCR_LANGUAGES = ['th', 'en']
    MIN_AMOUNT = 0.01
    MAX_AMOUNT = 10_000_000.00
    MIN_CONFIDENCE_THRESHOLD = 100  # Minimum score to stop early
    MAX_PREPROCESSING_VARIANTS = 3  # Reduce from 5 to 3 for performance


class SlipOCRProcessor:
    """OCR + NLP processor for Thai bank slips with flexible input handling."""

    def __init__(self, use_gpu: bool = False, config: SlipConfig = None):
        logger.info("⏳ Loading EasyOCR model (th,en)...")
        self.reader = easyocr.Reader(SlipConfig.OCR_LANGUAGES, gpu=use_gpu)
        self.config = config or SlipConfig()
        
        self.tokenizer = Tokenizer(custom_dict=self.config.BANK_KEYWORDS)
        logger.info("✅ Ready!")

    # ---------- Image I/O & preprocessing ----------
    def _load_image(self, source: ImgLike) -> np.ndarray:
        """
        Load image from various sources into BGR numpy array for OpenCV.
        - str path: handles common formats; PDF -> first page; HEIC via pillow-heif if available
        - bytes: decoded via PIL if possible, else cv2.imdecode
        - PIL.Image.Image: converted to BGR
        - np.ndarray: accepted; convert to BGR if grayscale/RGB
        """
        if isinstance(source, np.ndarray):
            img = source
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 3:
                # assume RGB, convert to BGR
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            return img

        if Image is not None and isinstance(source, Image.Image):
            pil = self._autorotate_exif(source)
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        if isinstance(source, (bytes, bytearray)):
            if Image is not None:
                try:
                    pil = Image.open(io.BytesIO(source))
                    pil = self._autorotate_exif(pil)
                    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"PIL decode failed: {e}, trying cv2")
            # fallback to cv2.imdecode
            data = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Cannot decode image bytes")
            return img

        if isinstance(source, str):
            path = source
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Path not found: {path}")

            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                if Image is None:
                    raise RuntimeError("PIL not available for PDF pipeline")
                pil = _pdf_to_image_first_page(path)
                if pil is None:
                    raise RuntimeError("Failed to render first page from PDF. Install pdf2image + poppler.")
                pil = self._autorotate_exif(pil)
                return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            if Image is not None:
                try:
                    pil = Image.open(path)
                    pil = self._autorotate_exif(pil)
                    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"PIL open failed: {e}, trying cv2")

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot read image: {path}")
            return img

        raise TypeError("Unsupported image source type. Provide path/bytes/PIL/ndarray.")

    def _autorotate_exif(self, pil_img: "Image.Image") -> "Image.Image":
        """Respect EXIF orientation if present"""
        try:
            if hasattr(ImageOps, "exif_transpose"):
                pil_img = ImageOps.exif_transpose(pil_img)
        except Exception as e:
            logger.warning(f"EXIF rotation failed: {e}")
        return pil_img.convert("RGB")

    def _fast_deskew(self, gray: np.ndarray) -> np.ndarray:
        """Approximate deskew using minAreaRect on edges; returns rotated grayscale image."""
        try:
            edges = cv2.Canny(gray, 50, 150)
            coords = np.column_stack(np.where(edges > 0))
            if coords.size < 10:
                return gray
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Skip if angle is too small
            if abs(angle) < 0.5:
                return gray
                
            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return gray

    def _preprocess(self, bgr: np.ndarray) -> List[np.ndarray]:
        """
        Return a list of preprocessed grayscale images to try with OCR.
        Reduced to 3 variants for better performance.
        """
        # to gray
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # denoise (bilateral is good at preserving edges)
        den = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        # deskew
        des = self._fast_deskew(den)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enh = clahe.apply(des)

        # Best 3 variants
        otsu = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        adap_gauss = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)

        return [enh, otsu, adap_gauss]

    def _ocr_best_rotation(self, img_variants: List[np.ndarray]) -> Tuple[str, float]:
        """
        Try rotations 0/90/180/270 and choose the text with maximum alphanumeric length.
        Returns (best_text, average_confidence)
        """
        best_text = ""
        best_score = -1
        best_confidence = 0.0

        for g in img_variants:
            for angle in [0, 90, 180, 270]:
                try:
                    if angle != 0:
                        g_rot = cv2.rotate(g, {
                            90: cv2.ROTATE_90_CLOCKWISE,
                            180: cv2.ROTATE_180,
                            270: cv2.ROTATE_90_COUNTERCLOCKWISE
                        }[angle])
                    else:
                        g_rot = g

                    # EasyOCR with detail to get confidence
                    results = self.reader.readtext(g_rot, detail=1, paragraph=False)
                    
                    if not results:
                        continue
                    
                    # Extract text and calculate confidence
                    texts = [r[1] for r in results]
                    confidences = [r[2] for r in results]
                    text = "\n".join(texts).strip()
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0

                    score = sum(ch.isalnum() for ch in text) + text.count("\n") * 2
                    # Boost score by confidence
                    score = score * (0.7 + 0.3 * avg_conf)
                    
                    if score > best_score:
                        best_score = score
                        best_text = text
                        best_confidence = avg_conf

                    # Early exit if good enough
                    if best_score > self.config.MIN_CONFIDENCE_THRESHOLD:
                        logger.info(f"Early exit: score={best_score:.1f}, conf={avg_conf:.2f}")
                        return best_text, best_confidence

                except Exception as e:
                    logger.warning(f"OCR failed at {angle}°: {e}")
                    continue

        if not best_text:
            logger.warning("⚠️ No text could be extracted from image")
        
        return best_text, best_confidence

    # ---------- Validation methods ----------
    def validate_amount(self, amount: Optional[float]) -> bool:
        """Validate if amount is within reasonable range"""
        if amount is None:
            return False
        return self.config.MIN_AMOUNT <= amount <= self.config.MAX_AMOUNT

    def validate_account_number(self, account: Optional[str]) -> bool:
        """Basic validation for account number format"""
        if not account:
            return False
        # Remove formatting
        clean = account.replace("-", "").replace(" ", "").replace("x", "").replace("X", "")
        # Should have at least some digits
        return len(clean) >= 5 and any(c.isdigit() for c in clean)

    # ---------- NLP helpers ----------
    def tokenize_thai(self, text: str) -> List[str]:
        """Tokenize Thai text using pythainlp"""
        tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
        return [t for t in tokens if t.strip()]

    def extract_amount(self, text: str) -> Optional[float]:
        """
        Extract transferred amount. Supports:
        - 1,234.56 บาท / ฿1,234.56 / 1234.56 / 1,234 / 1234
        Prioritizes lines near 'จำนวนเงิน' or 'ยอดเงิน'.
        """
        lines = text.splitlines()
        priority_blob = ""
        for i, line in enumerate(lines):
            if any(key in line for key in ['จำนวนเงิน', 'ยอดเงิน', 'Amount', 'ยอดโอน', 'โอน']):
                priority_blob = " ".join(lines[max(0, i-1):min(len(lines), i+3)])
                break

        patterns_priority = [
            r'(\d{1,3}(?:,\d{3})*\.\d{2})\s*บาท',
            r'฿\s*(\d{1,3}(?:,\d{3})*\.\d{2})',
            r'(\d{1,3}(?:,\d{3})*\.\d{2})\b',
            r'฿\s*(\d{1,3}(?:,\d{3})*)\b',
            r'(\d{1,3}(?:,\d{3})*)\s*บาท',
        ]
        
        for blob in [priority_blob, text]:
            if not blob:
                continue
            for pat in patterns_priority:
                matches = re.finditer(pat, blob)
                for m in matches:
                    s = m.group(1).replace(",", "")
                    try:
                        amount = float(s) if "." in s else float(int(s))
                        if self.validate_amount(amount):
                            return amount
                    except Exception:
                        continue
        return None

    def extract_date(self, text: str) -> Optional[str]:
        """Extract date from various Thai and international formats"""
        patterns = [
            r'(\d{4}-\d{1,2}-\d{1,2})',
            r'(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(\d{1,2}-\d{1,2}-\d{2,4})',
            r'(\d{1,2}\s*(?:ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.)\s*\d{2,4})'
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()
        return None

    def extract_time(self, text: str) -> Optional[str]:
        """Extract time in HH:MM or HH:MM:SS format"""
        m = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', text)
        return m.group(1) if m else None

    def extract_account_numbers(self, text: str) -> Dict[str, Optional[str]]:
        """Extract account numbers with improved type detection"""
        pats = [
            r'\b\d{3}-\d{1}-\d{5}-\d{1}\b',
            r'\b\d{3}-\d{6}-\d{1}\b',
            r'\b\d{10,12}\b',
            r'\b[xX]{3}-[xX]{1}-[xX]{2}\d{3}-\d{1}\b',
            r'\b[xX]{3}-[xX]+\d+-\d+\b',
            r'\b[xX]{3}\s+[xX]{1}\s+[xX]{2}\d{3}-\d{1}\b',
            r'\b0[0-9]{9}\b',
            r'\b0[0-9]{1}-[0-9]{4}-[0-9]{4}\b',
        ]
        accs: List[str] = []
        for p in pats:
            accs += re.findall(p, text)

        # unique preserving order
        seen = set()
        uniq = []
        for a in accs:
            if a not in seen:
                seen.add(a)
                uniq.append(a)

        def _type(a: str) -> str:
            """Determine if account is PromptPay or bank account"""
            raw = a.replace("-", "").replace(" ", "").replace("x", "").replace("X", "")
            
            # PromptPay: 10-digit mobile number starting with 0
            if len(raw) == 10 and raw.isdigit() and raw.startswith("0"):
                # Check if it looks like a phone number (starts with 06, 08, 09)
                if raw[1] in ['6', '8', '9']:
                    return 'promptpay'
            
            # PromptPay: 13-digit national ID
            if len(raw) == 13 and raw.isdigit():
                return 'promptpay'
            
            return 'bank'

        res = {
            'sender': None, 
            'receiver': None, 
            'sender_type': None, 
            'receiver_type': None
        }
        
        if len(uniq) >= 1:
            res['sender'] = uniq[0]
            res['sender_type'] = _type(uniq[0])
        if len(uniq) >= 2:
            res['receiver'] = uniq[1]
            res['receiver_type'] = _type(uniq[1])
            
        return res

    def extract_reference_number(self, text: str) -> Optional[str]:
        """Extract reference/transaction number"""
        patterns = [
            r'เลขที่อ้างอิง[:\s]*([A-Za-z0-9\-]+)',
            r'Reference[:\s]*([A-Za-z0-9\-]+)',
            r'Ref\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)',
            r'อ้างอิง[:\s]*([A-Za-z0-9\-]+)',
            r'Trans(?:action)?\.?\s*(?:ID|No|#)[:\s]*([A-Za-z0-9\-]+)'
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                ref = m.group(1).strip()
                # Filter out too short or too long references
                if 5 <= len(ref) <= 30:
                    return ref
        return None

    def extract_bank_names(self, text: str) -> Dict[str, Optional[str]]:
        """Extract bank names and codes"""
        found = []
        for name, code in self.config.BANK_NAMES.items():
            if name in text:
                found.append((name, code))
        
        res = {
            'sender_bank': None, 
            'receiver_bank': None,
            'sender_bank_code': None,
            'receiver_bank_code': None
        }
        
        if found:
            res['sender_bank'] = found[0][0]
            res['sender_bank_code'] = found[0][1]
            
            if len(found) > 1:
                res['receiver_bank'] = found[1][0]
                res['receiver_bank_code'] = found[1][1]
            else:
                res['receiver_bank'] = found[0][0]
                res['receiver_bank_code'] = found[0][1]
                
        return res

    def extract_names(self, text: str) -> Dict[str, Optional[str]]:
        """Extract sender and receiver names with length limits"""
        res = {'sender_name': None, 'receiver_name': None}
        
        # Sender patterns
        sender_patterns = [
            r'จาก\s+([ก-๙A-Za-z\s\.]{2,50}?)(?:\s+[xX#]|\s+ธนาคาร|\s+\d{3}|\n)',
            r'ผู้โอน[:\s]*([ก-๙A-Za-z\s\.]{2,50}?)(?:\s+ธนาคาร|\s+\d{3}|\n)',
            r'From[:\s]*([A-Za-z\s\.]{2,50}?)(?:\s+Bank|\s+\d{3}|\n)',
        ]
        
        for pat in sender_patterns:
            m = re.search(pat, text)
            if m:
                name = m.group(1).strip()
                # Remove trailing numbers/special chars
                name = re.sub(r'[\d\*\#]+$', '', name).strip()
                if len(name) >= 2:
                    res['sender_name'] = name
                    break

        # Receiver patterns
        receiver_patterns = [
            r'ไปยัง\s+([ก-๙A-Za-z\s\.]{2,50}?)(?:\s+พร้อมเพย์|\s+ธนาคาร|\s+\d{3}|\n)',
            r'ผู้รับ[:\s]*([ก-๙A-Za-z\s\.]{2,50}?)(?:\s+ธนาคาร|\s+\d{3}|\n)',
            r'ถึง[:\s]*([ก-๙A-Za-z\s\.]{2,50}?)(?:\s+ธนาคาร|\s+\d{3}|\n)',
            r'To[:\s]*([A-Za-z\s\.]{2,50}?)(?:\s+Bank|\s+\d{3}|\n)',
        ]
        
        for pat in receiver_patterns:
            m = re.search(pat, text)
            if m:
                name = m.group(1).strip()
                # Remove trailing numbers/special chars
                name = re.sub(r'[\d\*\#]+$', '', name).strip()
                if len(name) >= 2:
                    res['receiver_name'] = name
                    break
                    
        return res

    def extract_fee(self, text: str) -> Optional[float]:
        """Extract transaction fee"""
        patterns = [
            r'ค่าธรรมเนียม[:\s]*([\d,]+(?:\.\d{1,2})?)',
            r'Fee[:\s]*([\d,]+(?:\.\d{1,2})?)',
            r'ค่าใช้จ่าย[:\s]*([\d,]+(?:\.\d{1,2})?)',
        ]
        
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                try:
                    fee = float(m.group(1).replace(",", ""))
                    if 0 <= fee <= 1000:  # Reasonable fee range
                        return fee
                except Exception:
                    continue
        return None

    # ---------- Public APIs ----------
    def extract_text_from_slip(self, source: ImgLike, preprocess: bool = True) -> Tuple[str, float]:
        """
        Extract text from slip image.
        Returns: (text, confidence_score)
        """
        bgr = self._load_image(source)
        
        if preprocess:
            variants = self._preprocess(bgr)
            text, confidence = self._ocr_best_rotation(variants)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            text, confidence = self._ocr_best_rotation([gray])
            
        return text, confidence

    def summarize_slip(self, text: str, confidence: float = 0.0) -> Dict:
        """Summarize slip information with validation"""
        tokens = self.tokenize_thai(text)
        
        amount = self.extract_amount(text)
        accounts = self.extract_account_numbers(text)
        
        summary = {
            'raw_text': text,
            'ocr_confidence': round(confidence, 3),
            'tokens_count': len(tokens),
            'tokens_sample': tokens[:20],
            'amount': amount,
            'amount_valid': self.validate_amount(amount),
            'date': self.extract_date(text),
            'time': self.extract_time(text),
            'accounts': accounts,
            'reference': self.extract_reference_number(text),
            'banks': self.extract_bank_names(text),
            'names': self.extract_names(text),
            'fee': self.extract_fee(text),
            'processed_at': datetime.now().isoformat()
        }
        
        return summary

    def format_to_custom_json(self, summary: Dict) -> Dict:
        """Convert to custom JSON format"""
        menu_items = []
        
        # Add fee if exists
        if summary.get('fee'):
            menu_items.append({
                "type": "ค่าธรรมเนียม",
                "name": "ค่าธรรมเนียมการโอน",
                "price": float(summary['fee'])
            })
        
        # Add main transfer amount
        if summary['amount']:
            receiver_name = summary['names'].get('receiver_name') or 'ผู้รับ'
            menu_items.append({
                "type": "โอนเงิน",
                "name": f"โอนเงินถึง {receiver_name}",
                "price": float(summary['amount'])
            })
        
        total_price = sum(m['price'] for m in menu_items)
        total_items = len(menu_items)

        # Create timestamp
        time_str = None
        if summary.get('date') and summary.get('time'):
            time_str = f"{summary['date']} {summary['time']}"

        custom = {
            "payment": summary['banks'].get('sender_bank') or "ไม่ระบุธนาคาร",
            "name": summary['names'].get('sender_name') or "ไม่ระบุชื่อ",
            "money": f"{summary['amount']:.2f}" if summary['amount'] is not None else "0.00",
            "time": time_str or summary.get('processed_at'),
            "menu": menu_items,
            "total_items": total_items,
            "total_price": round(total_price, 2),
            "transfer_details": {
                "receiver_name": summary['names'].get('receiver_name'),
                "receiver_bank": summary['banks'].get('receiver_bank'),
                "receiver_bank_code": summary['banks'].get('receiver_bank_code'),
                "receiver_account": summary['accounts'].get('receiver'),
                "receiver_account_type": summary['accounts'].get('receiver_type'),
                "sender_account": summary['accounts'].get('sender'),
                "sender_account_type": summary['accounts'].get('sender_type'),
                "reference": summary.get('reference'),
                "fee": summary.get('fee')
            },
            "validation": {
                "amount_valid": summary.get('amount_valid', False),
                "ocr_confidence": summary.get('ocr_confidence', 0.0)
            }
        }
        return custom

    def process_slip(
        self, 
        source: ImgLike, 
        preprocess: bool = True, 
        output_format: Literal['standard', 'custom'] = 'standard'
    ) -> Dict:
        """
        Main processing pipeline
        
        Args:
            source: Image source (path, bytes, numpy array, PIL Image)
            preprocess: Enable preprocessing pipeline
            output_format: 'standard' or 'custom' JSON format
            
        Returns:
            Dictionary with extracted slip information
        """
        logger.info("🔍 Starting OCR processing...")
        text, confidence = self.extract_text_from_slip(source, preprocess=preprocess)
        logger.info(f"📝 OCR completed: {len(text)} chars, confidence: {confidence:.2%}")

        logger.info("✂️ Tokenizing & summarizing...")
        summary = self.summarize_slip(text, confidence)

        if output_format == 'custom':
            return self.format_to_custom_json(summary)
        return summary

    def print_summary(self, summary: Dict):
        """Pretty print slip summary"""
        print("\n" + "="*60)
        print("📊 Slip Summary")
        print("="*60)
        
        # OCR Quality
        if 'ocr_confidence' in summary:
            print(f"\n🎯 OCR Confidence: {summary['ocr_confidence']:.1%}")
        
        # Amount
        amount = summary.get('amount', 'ไม่พบ')
        amount_str = f"{amount:,.2f}" if isinstance(amount, (int, float)) else amount
        valid_flag = "✓" if summary.get('amount_valid') else "⚠"
        print(f"💰 Amount: {amount_str} บาท {valid_flag}")
        
        # Date & Time
        print(f"📅 Date: {summary.get('date', 'ไม่พบ')}")
        print(f"⏰ Time: {summary.get('time', 'ไม่พบ')}")
        
        # Fee
        if summary.get('fee'):
            print(f"💳 Fee: {summary['fee']:.2f} บาท")

        # Banks
        banks = summary.get('banks', {})
        sb = banks.get('sender_bank') or 'ไม่พบ'
        sc = f" ({banks.get('sender_bank_code')})" if banks.get('sender_bank_code') else ''
        print(f"\n🏦 Sender Bank: {sb}{sc}")
        
        rb = banks.get('receiver_bank') or 'ไม่พบ'
        rc = f" ({banks.get('receiver_bank_code')})" if banks.get('receiver_bank_code') else ''
        print(f"🏦 Receiver Bank: {rb}{rc}")

        # Names & Accounts
        names = summary.get('names', {})
        accs = summary.get('accounts', {})
        
        print(f"\n👤 Sender: {names.get('sender_name') or 'ไม่พบ'}")
        sa = accs.get('sender') or 'ไม่พบ'
        st = f" [{accs.get('sender_type', '')}]" if accs.get('sender_type') else ''
        print(f"   Account: {sa}{st}")

        print(f"\n👤 Receiver: {names.get('receiver_name') or 'ไม่พบ'}")
        ra = accs.get('receiver') or 'ไม่พบ'
        rt = f" [{accs.get('receiver_type', '')}]" if accs.get('receiver_type') else ''
        print(f"   Account: {ra}{rt}")

        print(f"\n🔖 Reference: {summary.get('reference') or 'ไม่พบ'}")
        print(f"\n✂️ Tokens: {summary.get('tokens_count', 0)}")
        
        sample = summary.get('tokens_sample', [])
        if sample:
            print(f"🔍 Sample: {', '.join(sample[:10])}")

        # Raw text preview
        raw = summary.get('raw_text', '')
        if raw:
            print("\n" + "-"*60)
            print("📄 OCR Raw Text Preview:")
            print("-"*60)
            preview = raw[:800]
            print(preview + ("..." if len(raw) > 800 else ""))
        
        print("\n" + "="*60)


def save_result(result: Dict, output_format: str = 'custom', output_dir: str = '.') -> str:
    """
    Save result to JSON file with timestamp and reference
    
    Args:
        result: Dictionary to save
        output_format: 'standard' or 'custom'
        output_dir: Directory to save file
        
    Returns:
        Filename of saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ref = ""
    
    # Extract reference for filename
    if output_format == 'custom':
        ref_val = result.get('transfer_details', {}).get('reference')
    else:
        ref_val = result.get('reference')
    
    if ref_val:
        # Clean reference for filename
        ref_clean = re.sub(r'[^\w\-]', '', str(ref_val)[:12])
        ref = f"_{ref_clean}"
    
    filename = f"slip_{output_format}_{ts}{ref}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Saved: {filepath}")
    return filepath


def batch_process_slips(
    processor: SlipOCRProcessor,
    input_paths: List[str],
    output_format: str = 'custom',
    output_dir: str = 'output',
    preprocess: bool = True
) -> List[Dict]:
    """
    Process multiple slips in batch
    
    Args:
        processor: SlipOCRProcessor instance
        input_paths: List of image file paths
        output_format: 'standard' or 'custom'
        output_dir: Directory to save results
        preprocess: Enable preprocessing
        
    Returns:
        List of results
    """
    results = []
    
    logger.info(f"📦 Batch processing {len(input_paths)} slips...")
    
    for i, path in enumerate(input_paths, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {i}/{len(input_paths)}: {path}")
        logger.info(f"{'='*60}")
        
        try:
            result = processor.process_slip(
                path, 
                preprocess=preprocess,
                output_format=output_format
            )
            results.append({
                'file': path,
                'status': 'success',
                'data': result
            })
            
            # Save individual result
            save_result(result, output_format=output_format, output_dir=output_dir)
            
        except Exception as e:
            logger.error(f"❌ Failed to process {path}: {e}")
            results.append({
                'file': path,
                'status': 'error',
                'error': str(e)
            })
    
    # Save batch summary
    summary_file = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total': len(input_paths),
            'success': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'error'),
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n📊 Batch summary saved: {summary_file}")
    return results


def main_demo():
    """Main entry point with CLI arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Universal Slip OCR - Extract information from Thai bank transfer slips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single slip with custom format
  python slip_ocr_processor.py slip.jpg --custom
  
  # Process without preprocessing
  python slip_ocr_processor.py slip.png --no-pre
  
  # Batch process multiple slips
  python slip_ocr_processor.py slip1.jpg slip2.png slip3.pdf --batch --custom
  
  # Specify output directory
  python slip_ocr_processor.py slip.jpg --output ./results
        """
    )
    
    parser.add_argument(
        "input", 
        nargs='+',
        help="Image/PDF path(s)"
    )
    parser.add_argument(
        "--custom", 
        action="store_true", 
        help="Output custom JSON format"
    )
    parser.add_argument(
        "--no-pre", 
        action="store_true", 
        help="Disable preprocessing"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch process multiple files"
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for OCR if available"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()

    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # GPU check
    use_gpu = args.gpu
    if use_gpu:
        try:
            import torch
            use_gpu = torch.cuda.is_available()
            if not use_gpu:
                logger.warning("GPU requested but CUDA not available, using CPU")
        except ImportError:
            logger.warning("PyTorch not installed, using CPU")
            use_gpu = False
    
    logger.info(f"🖥️ GPU Mode: {'Enabled' if use_gpu else 'Disabled'}")

    # Initialize processor
    proc = SlipOCRProcessor(use_gpu=use_gpu)

    fmt = 'custom' if args.custom else 'standard'
    
    # Batch or single processing
    if args.batch or len(args.input) > 1:
        results = batch_process_slips(
            proc,
            args.input,
            output_format=fmt,
            output_dir=args.output,
            preprocess=not args.no_pre
        )
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"\n✅ Batch complete: {success_count}/{len(args.input)} successful")
        
    else:
        # Single file processing
        result = proc.process_slip(
            args.input[0], 
            preprocess=not args.no_pre, 
            output_format=fmt
        )
        proc.print_summary(result)
        save_result(result, output_format=fmt, output_dir=args.output)


if __name__ == "__main__":
    main_demo()