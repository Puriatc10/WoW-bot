"""
humanize.py
------------
تزریق تصادفی‌بودن و بی‌دقتی طبیعی به تایمینگ و اکشن‌های بات، تا الگوی رفتاری
کمتر "ماشینی" و تکراری به نظر برسه.

مهم: این‌ها فقط ریسک تشخیص رو کم می‌کنن، نه صفر. هیچ تضمینی وجود نداره.
"""

import random
import time


def jitter(base: float, pct: float = 0.25, min_val: float = 0.0) -> float:
    """
    یه مقدار پایه رو با درصد مشخصی نویز تصادفی می‌ده.
    مثلاً jitter(1.5, 0.3) چیزی بین ۱.۰۵ تا ۱.۹۵ برمی‌گردونه.
    """
    delta = base * pct
    value = base + random.uniform(-delta, delta)
    return max(min_val, value)


def human_reaction_delay(base: float = 0.25, pct: float = 0.6) -> float:
    """
    تاخیر واکنش قبل از اکشن (مثلاً بعد از دیدن دشمن، قبل از Tab زدن).
    انسان‌ها معمولاً ۱۵۰-۴۰۰ میلی‌ثانیه واکنش نشون می‌دن، نه صفر.
    """
    return jitter(base, pct, min_val=0.08)


def should_take_micro_break(probability: float = 0.02) -> bool:
    """
    هر تیک یه احتمال کوچیک برای یه مکث کوتاه (انگار داره اطراف رو نگاه می‌کنه
    یا حواسش پرت شده) - باعث می‌شه الگوی حرکت کاملاً یکنواخت نباشه.
    """
    return random.random() < probability


def micro_break_duration() -> float:
    return random.uniform(0.4, 2.5)


def should_take_long_break(session_duration_sec: float, avg_interval_sec: float = 25 * 60) -> bool:
    """
    شبیه‌سازی استراحت‌های طولانی‌تر (مثل رفتن AFK چندلحظه‌ای).
    احتمالش با گذشت زمون تجمع پیدا می‌کنه (پواسون-مانند ساده).
    """
    # احتمال تقریبی در هر تیک که منجر به میانگین یک بار در avg_interval_sec بشه
    tick_hz = 10.0
    p_per_tick = 1.0 / (avg_interval_sec * tick_hz)
    return random.random() < p_per_tick


def long_break_duration() -> float:
    return random.uniform(15, 90)  # ۱۵ تا ۹۰ ثانیه، شبیه چک کردن گوشی وسط فارم


def humanized_rotation_gcd(base_gcd: float = 1.5) -> float:
    """
    به‌جای فاصله‌ی دقیق GCD، یه تاخیر واکنشی روش اضافه می‌کنه (انسان‌ها معمولاً
    کمی دیرتر از لحظه‌ی دقیق GCD اسکیل بعدی رو می‌زنن، نه زودتر).
    """
    return base_gcd + human_reaction_delay(base=0.15, pct=0.8)


def humanized_key_hold() -> float:
    """مدت نگه‌داشتن کلید برای شبیه‌سازی فشار واقعی (نه دقیقاً ثابت)."""
    return jitter(0.06, 0.5, min_val=0.02)


def humanized_turn_tolerance(base: float = 0.08) -> float:
    """انسان‌ها دقیقاً روی هدف نمی‌ایستن؛ یه کم تلورانس بیشتر/متغیر طبیعی‌تره."""
    return jitter(base, 0.4, min_val=0.04)


class SessionClock:
    """برای ردیابی مدت‌زمان سشن فعلی، جهت تصمیم‌گیری استراحت‌های بلند."""

    def __init__(self):
        self.started_at = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at
