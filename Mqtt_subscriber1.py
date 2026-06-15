

import json
import paho.mqtt.client as mqtt

BROKER = "localhost"
TOPIC  = "smart_irrigation/sensors"

def on_connect(client, userdata, flags, rc):
    print(f"[✔] Connected to MQTT Broker (rc={rc})")
    client.subscribe(TOPIC)
    print(f"[✔] Subscribed to topic: {TOPIC}\n")

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    print("─" * 45)
    print("[📡 Received IoT Sensor Data]")
    for k, v in payload.items():
        print(f"  {k:<30} : {v}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.loop_forever()
