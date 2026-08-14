
# ============================================================
# rule_based_baseline.py
# Rule-Based Controller for Health Monitoring Environment
# اقدامات: 0=مسیر عادی, 1=هشدار اولویت بالا, 
#          2=کاهش نرخ نمونه‌برداری, 3=تغییر مسیر به پشتیبان
# ============================================================

import numpy as np

class RuleBasedBaseline:
    """
    Rule-based decision maker for IoMT health monitoring.
    Rules are based on:
        - predicted probability of critical event (from LSTM)
        - battery level
        - network congestion
    """
    def __init__(self):
        # آستانه‌ها (قابل تنظیم)
        self.pred_threshold_high = 0.7      # برای اقدام ۱ (هشدار)
        self.pred_threshold_low = 0.3       # برای اقدام ۰ (مسیر عادی)
        self.battery_threshold_low = 20.0   # درصد
        self.congestion_threshold_high = 0.8

        # وزن‌ها برای ترکیب در صورت تداخل قوانین
        self.pred_weight = 2.0
        self.battery_weight = 1.5
        self.congestion_weight = 1.0

    def get_action(self, predicted_prob, battery_level, network_congestion):
        """
        ورودی:
            predicted_prob: float [0,1] – احتمال بحرانی بودن وضعیت بیمار
            battery_level: float [0,100] – درصد باتری باقی‌مانده
            network_congestion: float [0,1] – میزان ازدحام شبکه
        خروجی:
            action: int در {0,1,2,3}
        """
        # ============================================================
        # 1. قانون اصلی: اولویت با بحران
        # ============================================================
        if predicted_prob >= self.pred_threshold_high:
            # بحران قریب‌الوقوع → هشدار اولویت بالا
            action = 1
            # اگر باتری خیلی کم است، به‌جای هشدار، کاهش نرخ نمونه‌برداری انجام شود
            if battery_level < self.battery_threshold_low:
                action = 2
            return action

        # ============================================================
        # 2. قوانین مربوط به باتری و ازدحام (در شرایط غیربحرانی)
        # ============================================================
        # اگر باتری کم است و ازدحام زیاد نیست → کاهش نرخ نمونه‌برداری
        if battery_level < self.battery_threshold_low:
            if network_congestion < self.congestion_threshold_high:
                return 2  # کاهش نرخ نمونه‌برداری
            else:
                # هم باتری کم و هم ازدحام زیاد → تغییر مسیر به پشتیبان
                return 3

        # اگر ازدحام بالا است و باتری مناسب است → تغییر مسیر به پشتیبان
        if network_congestion >= self.congestion_threshold_high:
            if predicted_prob < self.pred_threshold_low:
                return 3  # فقط در صورتی که بحران هم نباشد
            else:
                # ازدحام بالا و احتمال بحران متوسط → هشدار یا مسیر عادی؟
                # اینجا هشدار می‌دهیم چون احتمال بحران متوسط است
                return 1

        # ============================================================
        # 3. حالت پیش‌فرض: مسیر عادی
        # ============================================================
        return 0

    def explain(self, predicted_prob, battery_level, network_congestion):
        """
        توضیح منطق انتخاب اقدام (برای گزارش‌گیری)
        """
        action = self.get_action(predicted_prob, battery_level, network_congestion)
        reasons = {
            0: "No critical condition, battery and network are in good state.",
            1: f"Critical event predicted (prob={predicted_prob:.2f}) or moderate risk with high congestion.",
            2: f"Low battery ({battery_level:.1f}%) and network is acceptable, reducing sampling rate.",
            3: f"High network congestion ({network_congestion:.2f}) with no critical risk, rerouting via backup."
        }
        return action, reasons[action]
