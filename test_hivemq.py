# test_hivemq.py
import paho.mqtt.client as mqtt
import json, time, math, random, ssl

HOST = "79234fef6e85454480c0ce543e7dcecb.s1.eu.hivemq.cloud"
PORT = 8883
USER = "Issra"
PASS = "Issra2026"

c = mqtt.Client()
c.username_pw_set(USER, PASS)
c.tls_set(cert_reqs=ssl.CERT_NONE)
c.tls_insecure_set(True)
c.connect(HOST, PORT)
c.loop_start()

print("Envoi données test vers HiveMQ...")
t = 0
while True:
    val = int(2048 + 500*math.sin(2*math.pi*1.2*t) + 30*random.gauss(0,1))
    c.publish("ecg/data", json.dumps({
        "c1": val, "c2": val, "c3": val,
        "t": int(t*1000)
    }))
    t += 1/250.0
    time.sleep(1/250.0)