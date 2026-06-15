

import json
import queue
import threading
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT   = 1883
TOPIC  = "smart_irrigation/sensors"

# shared thread-safe queue (maxsize=1 → always latest reading)
mqtt_queue: queue.Queue = queue.Queue(maxsize=1)


class MQTTBridge:
    """
    Lightweight MQTT listener that runs in a daemon thread.
    Puts the latest sensor payload into `mqtt_queue`.
    """

    def __init__(self, broker: str = BROKER, port: int = PORT, topic: str = TOPIC):
        self.broker = broker
        self.port   = port
        self.topic  = topic
        self._client = mqtt.Client()
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    # ── callbacks ─────────────────────────────────────────────
    def _on_connect(self, client, userdata, flags, rc):
        client.subscribe(self.topic)
        print(f"[MQTTBridge] Connected & subscribed to '{self.topic}'")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            # discard old value so queue never blocks
            if mqtt_queue.full():
                try:
                    mqtt_queue.get_nowait()
                except queue.Empty:
                    pass
            mqtt_queue.put_nowait(payload)
        except Exception as e:
            print(f"[MQTTBridge] Parse error: {e}")

    # ── public API ────────────────────────────────────────────
    def start(self):
        """Connect and start loop in a background daemon thread."""
        try:
            self._client.connect(self.broker, self.port, 60)
            t = threading.Thread(target=self._client.loop_forever, daemon=True)
            t.start()
            print("[MQTTBridge] Background thread started")
        except Exception as e:
            print(f"[MQTTBridge] Could not connect to broker: {e}")

    def get(self) -> dict | None:
        """
        Return the latest sensor reading, or None if nothing received yet.
        Non-blocking.
        """
        try:
            return mqtt_queue.get_nowait()
        except queue.Empty:
            return None
