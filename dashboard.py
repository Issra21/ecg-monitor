import joblib, json, threading, time, ssl, os
import smtplib
import numpy as np, pandas as pd
import paho.mqtt.client as mqtt
from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dash import Dash, dcc, html, Output, Input, callback_context
import plotly.graph_objects as go
from pipeline_ecg import ecg_filt, get_rp, get_rr, qc, hrv
import sys

print("=== DÉMARRAGE DASHBOARD CLOUD ===", flush=True)

# ══════════════════════════════════════════
# ── CONFIGURATION (Ajustée pour le Cloud) ──
# ══════════════════════════════════════════
# Utilisez vos identifiants EMQX Cloud
HIVEMQ_HOST = "ca3c1512.ala.eu-central-1.emqxsl.com"
HIVEMQ_PORT = 8883
HIVEMQ_USER = "Issra"
HIVEMQ_PASS = "Issra2026"

# Email & SMS (Inchangés)
EMAIL_FROM = "issrasaidi13@gmail.com"
EMAIL_PASS = "lolo kccm apnu uwyu"
EMAIL_TO   = ["issra.saidi@univgb.tn", "famille@gmail.com"]

TWILIO_ENABLED = True
TWILIO_SID     = "US6fb0de42a3a7cf3d31f2f4386de58fa4"
TWILIO_TOKEN   = "9NCRWZ95U5G4DFM28XKE9HL3"
TWILIO_FROM    = "+21694137899"
TWILIO_TO      = ["+216987895"] # Ajoutez vos numéros ici

# ══════════════════════════════════════════
# ── CHARGER MODÈLE ──
# ══════════════════════════════════════════
# Note : Sur Render, les fichiers doivent être à la racine du projet GitHub
mdl      = joblib.load("svm_epilepsie.pkl")
params   = json.load(open("params_epilepsie.json"))
bg_stats = json.load(open("bg_stats_epilepsie.json"))

FEATS    = params["FEATS"]
THR      = params["THR_GLOBAL"]
WIN_S    = params["WIN_S"]
N_CONSEC = params["N_CONSEC"]
COOLDOWN = params["COOLDOWN"]
FS       = params["FS"]
BUF_SIZE = int(WIN_S * FS)

PATIENT_ID   = list(bg_stats.keys())[0]
PATIENT_NAME = "Issra Saidi"

# ══════════════════════════════════════════
# ── BUFFERS & ÉTAT ──
# ══════════════════════════════════════════
buf_ch1   = deque(maxlen=BUF_SIZE)
hist_t    = deque(maxlen=5000)
hist_ecg  = deque(maxlen=5000)
hist_prob = deque(maxlen=500)
hist_al   = deque(maxlen=500)
lock      = threading.Lock()

consec      = 0
last_alert  = -np.inf
alert_count = 0
start_time  = time.time()

# ── FONCTIONS ALERTE (MQTT/Email/SMS identiques) ──
# [Gardez vos fonctions send_alert_email, send_alert_sms, publish_mqtt_alert telles quelles]

# ══════════════════════════════════════════
# ── MQTT AVEC TLS (CRUCIAL POUR EMQX) ──
# ══════════════════════════════════════════
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connecté à EMQX Cloud")
        client.subscribe("ecg/data")
    else:
        print(f"❌ Erreur connexion EMQX: {rc}")

def on_message(client, userdata, msg):
    try:
        d = json.loads(msg.payload)
        t_base = float(d.get("t", time.time()*1000)) / 1000.0
        if "b" in d:
            batch = d["b"]
            n = len(batch)
            with lock:
                for i, val in enumerate(batch):
                    t_sample = t_base - (n - 1 - i) / 250.0
                    buf_ch1.append(float(val))
                    hist_t.append(t_sample)
                    hist_ecg.append(float(val))
    except Exception as e:
        print(f"MQTT erreur: {e}")

mqtt_client = mqtt.Client(client_id="Render_Dashboard_Server")
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.username_pw_set(HIVEMQ_USER, HIVEMQ_PASS)

# Configuration TLS pour le port 8883
mqtt_client.tls_set(cert_reqs=ssl.CERT_NONE) 
mqtt_client.tls_insecure_set(True)

def mqtt_start():
    try:
        mqtt_client.connect(HIVEMQ_HOST, HIVEMQ_PORT, keepalive=60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"❌ Échec initial MQTT: {e}")

threading.Thread(target=mqtt_start, daemon=True).start()

# ══════════════════════════════════════════
# ── DASHBOARD DASH ──
# ══════════════════════════════════════════
app = Dash(__name__)
server = app.server # Indispensable pour Gunicorn sur Render

# [Copiez ici votre app.layout et vos @app.callback sans changement]

if __name__ == "__main__":
    # Render définit automatiquement la variable d'environnement PORT
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)