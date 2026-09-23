import cv2
import numpy as np
import pytesseract

# مسیر Tesseract (اگه ویندوز داری)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class BarReader:
    def __init__(self, hp_roi, mana_roi):
        """
        hp_roi:   (x, y, w, h) گوشه چپ-بالا + عرض + ارتفاع
        mana_roi: (x, y, w, h)
        """
        self.hp_roi = hp_roi
        self.mana_roi = mana_roi

    def _ratio(self, frame, roi):
        """
        درصد پر بودن نوار رو از روی رنگ‌های اشباع‌شده حساب می‌کنه.
        پیکسل‌های رنگی (سبز یا نارنجی) = پر
        پیکسل‌های تیره/خاکستری = خالی
        """
        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w, :3]  # BGR
        
        # تبدیل به HSV برای تشخیص راحت‌تر رنگ
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # پیکسل‌هایی که "رنگی و روشن" هستن = پُر
        # (S بالا یعنی اشباع، V بالا یعنی روشن)
        colored = (S > 80) & (V > 80)
        
        # برای هر ستون، اگه بیشتر پیکسل‌هاش رنگی بود، اون ستون "پر" حساب می‌شه
        col_filled = colored.mean(axis=0) > 0.3
        
        if col_filled.sum() == 0:
            return 0.0
        
        # از چپ‌ترین ستون پر تا راست‌ترین ستون پر = نوار واقعی
        idxs = np.where(col_filled)[0]
        filled_width = idxs.max() - idxs.min() + 1
        
        return min(filled_width / w, 1.0)

    def read_hp(self, frame):
        ratio = self._ratio(frame, self.hp_roi)
        return ratio, int(ratio * 100)

    def read_mana(self, frame):
        ratio = self._ratio(frame, self.mana_roi)
        return ratio, int(ratio * 100)