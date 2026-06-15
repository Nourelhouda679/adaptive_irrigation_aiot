import json
import time
import random
import paho.mqtt.client as mqtt
from datetime import datetime

BROKER = "localhost"
TOPIC  = "smart_irrigation/sensors"

# ── مراحل محاكاة الـ drift ─────────────────────────────────────
# كل مرحلة: (اسم، مدة بالثواني، perturbation)
DRIFT_PHASES = [
    {
        "name"       : "Normal",
        "duration_s" : 120,          # دقيقتان — بيانات طبيعية
        "temp_shift" : 0,
        "hum_shift"  : 0,
        "rain_mult"  : 1.0,
        "moist_shift": 0,
    },
    {
        "name"       : "Drought Drift",
        "duration_s" : 60,           # دقيقة — جفاف مصطنع (drift)
        "temp_shift" : +10,          # حرارة أعلى
        "hum_shift"  : -20,          # رطوبة أقل
        "rain_mult"  : 0.1,          # مطر شبه معدوم
        "moist_shift": -15,          # رطوبة تربة أقل
    },
    {
        "name"       : "Post-Drift Normal",
        "duration_s" : 120,          # عودة للطبيعي — نرى هل النموذج تكيّف
        "temp_shift" : 0,
        "hum_shift"  : 0,
        "rain_mult"  : 1.0,
        "moist_shift": 0,
    },
]

def generate_reading(phase: dict) -> dict:
    """توليد قراءة IoT واقعية مع تطبيق perturbation المرحلة."""

    ps = phase

    # نطاقات واقعية (يومية/حقلية)
    temp     = round(random.uniform(12.0, 42.0)  + ps["temp_shift"],  2)
    humidity = round(random.uniform(25.0, 95.0)  + ps["hum_shift"],   2)
    rainfall = round(random.uniform(0.0,  150.0) * ps["rain_mult"],   2)
    moisture = round(random.uniform(8.0,  65.0)  + ps["moist_shift"], 2)

    # تقييد القيم لتبقى في نطاق واقعي
    temp     = max(5.0,   min(50.0,  temp))
    humidity = max(10.0,  min(100.0, humidity))
    rainfall = max(0.0,   min(200.0, rainfall))
    moisture = max(5.0,   min(70.0,  moisture))

    return {
        "timestamp"               : datetime.now().isoformat(),
        "drift_phase"             : ps["name"],
        "Soil_Moisture"           : moisture,
        "Temperature_C"           : temp,
        "Rainfall_mm"             : rainfall,
        "Humidity"                : humidity,
        "Sunlight_Hours"          : round(random.uniform(4.0,  11.0),  2),
        "Wind_Speed_kmh"          : round(random.uniform(0.5,  20.0),  2),
        "Organic_Carbon"          : round(random.uniform(0.3,  1.6),   2),
        "Electrical_Conductivity" : round(random.uniform(0.1,  3.5),   2),
        "Soil_pH"                 : round(random.uniform(4.8,  8.2),   2),
        "Field_Area_hectare"      : round(random.uniform(0.3,  15.0),  2),
        "Previous_Irrigation_mm"  : round(random.uniform(0.02, 120.0), 2),
        "Soil_Type"               : random.choice(["Clay", "Loamy", "Sandy", "Silt"]),
        "Crop_Type"               : random.choice(["Cotton", "Maize", "Potato",
                                                   "Rice", "Sugarcane", "Wheat"]),
        "Region"                  : random.choice(["North", "South", "East",
                                                   "West", "Central"]),
        "Water_Source"            : random.choice(["Groundwater", "Rainwater",
                                                   "Reservoir", "River"]),
        "Season"                  : random.choice(["Kharif", "Rabi", "Zaid"]),
        "Crop_Growth_Stage"       : random.choice(["Flowering", "Harvest",
                                                   "Sowing", "Vegetative"]),
        "Mulching_Used"           : random.choice(["Yes", "No"]),
    }


def main():
    client = mqtt.Client()
    client.connect(BROKER, 1883, 60)

    print(f"[✔] Connected to MQTT broker at {BROKER}")
    print(f"[✔] Publishing to topic: {TOPIC}")
    print(f"[✔] Drift simulation: {len(DRIFT_PHASES)} phases")
    print("[✔] Press Ctrl+C to stop\n")

    phase_idx  = 0
    phase_start = time.time()

    while True:
        # تحديد المرحلة الحالية
        phase     = DRIFT_PHASES[phase_idx]
        elapsed   = time.time() - phase_start

        if elapsed >= phase["duration_s"]:
            phase_idx   = (phase_idx + 1) % len(DRIFT_PHASES)
            phase_start = time.time()
            phase       = DRIFT_PHASES[phase_idx]
            print(f"\n[🔄 PHASE CHANGE] → {phase['name']}\n")

        data = generate_reading(phase)
        client.publish(TOPIC, json.dumps(data))

        print(
            f"[{phase['name']:20s}] "
            f"Moisture={data['Soil_Moisture']:5.1f} | "
            f"Temp={data['Temperature_C']:5.1f} | "
            f"Rain={data['Rainfall_mm']:6.1f} | "
            f"Humidity={data['Humidity']:5.1f}"
        )

        time.sleep(2)


if __name__ == "__main__":
    main()
