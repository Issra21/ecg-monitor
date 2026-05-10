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

# ══════════════════════════════════════════
# ── CONFIG EMAIL ──
# ══════════════════════════════════════════
EMAIL_FROM = "issrasaidi13@gmail.com"
EMAIL_PASS = "lolo kccm apnu uwyu"
EMAIL_TO   = [
    "issra.saidi@univgb.tn",
    "famille@gmail.com",
]

# ══════════════════════════════════════════
# ── CONFIG SMS TWILIO ──
# ══════════════════════════════════════════
TWILIO_ENABLED = True
TWILIO_SID     = "US6fb0de42a3a7cf3d31f2f4386de58fa4"
TWILIO_TOKEN   = "9NCRWZ95U5G4DFM28XKE9HL3"
TWILIO_FROM    = "+21694137899"
TWILIO_TO      = [
    "+216987895",
    "+21622XXXXXX",
]

# ══════════════════════════════════════════
# ── CONFIG EMQX ──
# ══════════════════════════════════════════
HIVEMQ_HOST = "ca3c1512.ala.eu-central-1.emqxsl.com"
HIVEMQ_PORT = 8883
HIVEMQ_USER = "Issra"
HIVEMQ_PASS = "Issra2026"

# ══════════════════════════════════════════
# ── CHARGER MODÈLE ──
# ══════════════════════════════════════════
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
print(f"Patient : {PATIENT_NAME}")

# ══════════════════════════════════════════
# ── BUFFERS ──
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

# ══════════════════════════════════════════
# ── FONCTIONS ALERTE ──
# ══════════════════════════════════════════
def send_alert_email(proba, t_now):
    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_FROM
        msg["To"]      = ", ".join(EMAIL_TO)
        msg["Subject"] = "⚠ ALERTE CRISE EPILEPTIQUE — Issra Saidi"
        heure = time.strftime('%d/%m/%Y %H:%M:%S', time.localtime(t_now))
        body  = f"""
⚠ ALERTE MÉDICALE AUTOMATIQUE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Patient     : {PATIENT_NAME}
Événement   : Crise épileptique prédite
Probabilité : {proba*100:.1f}%
Date/Heure  : {heure}

🔗 Dashboard temps réel :
https://ecg-monitor-pxbt.onrender.com

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ce message est généré automatiquement
par le système de surveillance ECG IoT.
        """
        msg.attach(MIMEText(body, "plain", "utf-8"))
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASS)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        server.quit()
        print(f"✅ Email alerte envoyé à {EMAIL_TO}")
    except Exception as e:
        print(f"❌ Email erreur: {e}")


def send_alert_sms(proba, t_now):
    if not TWILIO_ENABLED:
        return
    try:
        from twilio.rest import Client as TwilioClient
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        heure  = time.strftime('%H:%M:%S', time.localtime(t_now))
        body   = (
            f"ALERTE CRISE - {PATIENT_NAME}\n"
            f"Probabilite: {proba*100:.1f}%\n"
            f"Heure: {heure}\n"
            f"Dashboard: https://ecg-monitor-pxbt.onrender.com"
        )
        for numero in TWILIO_TO:
            client.messages.create(body=body, from_=TWILIO_FROM, to=numero)
            print(f"✅ SMS alerte envoyé à {numero}")
    except Exception as e:
        print(f"❌ SMS erreur: {e}")


def publish_mqtt_alert(proba, t_now):
    try:
        alert_msg = json.dumps({
            "alert":   1,
            "patient": PATIENT_NAME,
            "proba":   round(proba, 3),
            "t":       t_now
        })
        mqtt_client.publish("ecg/alert", alert_msg)
        print("✅ Alerte MQTT publiée → ecg/alert")
    except Exception as e:
        print(f"❌ MQTT alert erreur: {e}")


# ══════════════════════════════════════════
# ── RAPPORT QUOTIDIEN 24H ──
# ══════════════════════════════════════════
def send_daily_report():
    global alert_count   # ← déclaré en premier, avant tout usage
    while True:
        time.sleep(86400)   # 24 heures
        try:
            heure   = time.strftime('%d/%m/%Y %H:%M:%S')
            uptime  = int(time.time() - start_time)
            upt_str = f"{uptime//3600:02d}h{(uptime%3600)//60:02d}min"

            with lock:
                n_al = alert_count

            # ── Email rapport ──
            msg = MIMEMultipart()
            msg["From"]    = EMAIL_FROM
            msg["To"]      = ", ".join(EMAIL_TO)
            msg["Subject"] = f"✅ Rapport quotidien — {PATIENT_NAME} — {time.strftime('%d/%m/%Y')}"
            body = f"""
✅ RAPPORT QUOTIDIEN — BONNE SANTÉ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Patient     : {PATIENT_NAME}
Date/Heure  : {heure}
Durée suivi : {upt_str}

État général : ✅ STABLE

Statistiques des dernières 24h :
  • Crises détectées : {n_al}
  • Système actif   : Oui
  • Acquisition ECG : En cours ({FS} Hz)

{"⚠ Attention : " + str(n_al) + " crise(s) détectée(s) aujourd'hui." if n_al > 0 else "✅ Aucune crise détectée — Patient en bonne santé."}

🔗 Dashboard temps réel :
https://ecg-monitor-pxbt.onrender.com

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rapport généré automatiquement
Système de surveillance ECG IoT
            """
            msg.attach(MIMEText(body, "plain", "utf-8"))
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
            server.quit()
            print("✅ Rapport quotidien email envoyé")

            # ── SMS rapport ──
            if TWILIO_ENABLED:
                from twilio.rest import Client as TwilioClient
                client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
                sms_body = (
                    f"✅ Rapport - {PATIENT_NAME}\n"
                    f"Date: {time.strftime('%d/%m/%Y')}\n"
                    f"Crises 24h: {n_al}\n"
                    f"{'⚠ Crises detectees!' if n_al > 0 else '✅ Bonne sante'}\n"
                    f"https://ecg-monitor-pxbt.onrender.com"
                )
                for numero in TWILIO_TO:
                    client.messages.create(body=sms_body, from_=TWILIO_FROM, to=numero)
                print("✅ Rapport quotidien SMS envoyé")

            # Reset compteur pour le prochain jour
            alert_count = 0

        except Exception as e:
            print(f"❌ Rapport quotidien erreur: {e}")


# ══════════════════════════════════════════
# ── MQTT AVEC RECONNEXION AUTOMATIQUE ──
# ══════════════════════════════════════════
def on_connect(client, userdata, flags, rc):
    print(f"EMQX connecté rc={rc}")
    if rc == 0:
        client.subscribe("ecg/data")
        print("✅ Abonné à ecg/data")
    else:
        print(f"❌ Connexion refusée rc={rc}")

def on_disconnect(client, userdata, rc):
    print(f"⚠ EMQX déconnecté rc={rc} — reconnexion automatique...")

def on_message(client, userdata, msg):
    try:
        d = json.loads(msg.payload)
        t_base = float(d["t"]) / 1000.0
        # Format batch : {"b":[v1,...,v25],"t":millis}
        if "b" in d:
            batch = d["b"]
            n = len(batch)
            with lock:
                for i, val in enumerate(batch):
                    t_sample = t_base - (n - 1 - i) / 250.0
                    buf_ch1.append(float(val))
                    hist_t.append(t_sample)
                    hist_ecg.append(float(val))
        # Format simple : {"c1":val,"t":millis}
        elif "c1" in d:
            with lock:
                buf_ch1.append(float(d["c1"]))
                hist_t.append(t_base)
                hist_ecg.append(float(d["c1"]))
    except Exception as e:
        print(f"MQTT message erreur: {e}")


def mqtt_connect():
    try:
        mqtt_client.username_pw_set(HIVEMQ_USER, HIVEMQ_PASS)
        mqtt_client.tls_set(cert_reqs=ssl.CERT_NONE)
        mqtt_client.tls_insecure_set(True)
        mqtt_client.connect(HIVEMQ_HOST, HIVEMQ_PORT, keepalive=60)
        mqtt_client.loop_start()
        print("EMQX connexion initiée...")
    except Exception as e:
        print(f"❌ EMQX erreur connexion: {e}")


def mqtt_watchdog():
    while True:
        time.sleep(30)
        try:
            if not mqtt_client.is_connected():
                print("⚠ EMQX déconnecté — reconnexion...")
                try:
                    mqtt_client.reconnect()
                except Exception:
                    mqtt_connect()
        except Exception as e:
            print(f"Watchdog erreur: {e}")


mqtt_client = mqtt.Client()
mqtt_client.on_connect    = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message    = on_message
mqtt_connect()
threading.Thread(target=mqtt_watchdog, daemon=True).start()


# ══════════════════════════════════════════
# ── THREAD INFÉRENCE ──
# ══════════════════════════════════════════
def run_inference():
    global consec, last_alert, alert_count
    while True:
        time.sleep(60)
        with lock:
            n = len(buf_ch1)
            if n < BUF_SIZE:
                print(f"Buffer {n}/{BUF_SIZE} — attente...")
                continue
            seg   = np.array(buf_ch1)
            t_now = list(hist_t)[-1] if hist_t else 0
        try:
            seg_f = ecg_filt(seg, FS)
            rp    = get_rp(seg_f, int(FS))
            rr    = get_rr(rp, FS)
            if not qc(rp, rr):
                print("QC échoué"); continue
            ft = hrv(rr)
            if ft is None: continue
            bs  = bg_stats.get(PATIENT_ID, {})
            med = bs.get("median", {})
            iqr = bs.get("iqr", {})
            fv  = pd.Series(ft)[FEATS].copy()
            for col in FEATS:
                m = med.get(col, 0); s = iqr.get(col, 1)
                fv[col] = (fv[col] - m) / (s if s > 1e-8 else 1)
            proba = float(mdl.predict_proba(
                fv.fillna(0).values.reshape(1,-1))[0,1])
            print(f"Proba={proba:.4f} seuil={THR:.4f}")
            consec = consec + 1 if proba >= THR else 0
            alert  = 0
            if consec >= N_CONSEC and (t_now - last_alert) >= COOLDOWN:
                alert = 1; last_alert = t_now
                consec = 0; alert_count += 1
                print(f"⚠ ALERTE CRISE #{alert_count}")
                threading.Thread(target=publish_mqtt_alert, args=(proba, t_now), daemon=True).start()
                threading.Thread(target=send_alert_email,   args=(proba, t_now), daemon=True).start()
                threading.Thread(target=send_alert_sms,     args=(proba, t_now), daemon=True).start()
            with lock:
                hist_prob.append((t_now, proba))
                hist_al.append((t_now, alert))
        except Exception as e:
            print(f"Inférence erreur: {e}")

threading.Thread(target=run_inference,    daemon=True).start()
threading.Thread(target=send_daily_report, daemon=True).start()

# ══════════════════════════════════════════
# ── COULEURS ──
# ══════════════════════════════════════════
BG    = "#0A0F1E"
CARD  = "#0D1528"
BDR   = "#1E2D4A"
BLUE  = "#00C8FF"
GREEN = "#00FF9C"
RED   = "#FF4560"
YEL   = "#FFB800"
MUTED = "#4A5880"
TEXT  = "#E0E8FF"

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="monospace", color=MUTED, size=10),
    margin=dict(l=48, r=12, t=8, b=32),
    showlegend=False,
    xaxis=dict(gridcolor=BDR, zerolinecolor=BDR),
    yaxis=dict(gridcolor=BDR, zerolinecolor=BDR),
)

CARD_STYLE = {
    "background": CARD, "border": f"1px solid {BDR}",
    "borderRadius": "10px", "padding": "14px",
}

# ══════════════════════════════════════════
# ── DASHBOARD ──
# ══════════════════════════════════════════
app    = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"background":BG,"minHeight":"100vh","fontFamily":"monospace","color":TEXT},
    children=[

    # Header
    html.Div(style={
        "display":"flex","alignItems":"center","justifyContent":"space-between",
        "padding":"14px 24px","borderBottom":f"1px solid {BDR}","background":CARD,
    }, children=[
        html.Div(style={"display":"flex","alignItems":"center","gap":"12px"}, children=[
            html.Span("♥", style={"fontSize":"28px","color":BLUE}),
            html.Div([
                html.Div("ECG MONITOR",
                         style={"fontSize":"18px","fontWeight":"700",
                                "color":BLUE,"letterSpacing":"2px"}),
                html.Div(f"Patient : {PATIENT_NAME}  —  Prédiction crises épileptiques",
                         style={"fontSize":"11px","color":MUTED}),
            ]),
        ]),
        html.Div(style={"display":"flex","alignItems":"center","gap":"8px"}, children=[
            html.Div(style={"width":"10px","height":"10px",
                            "borderRadius":"50%","background":GREEN}),
            html.Span("ACQUISITION EN COURS",
                      style={"fontSize":"11px","color":GREEN}),
        ]),
    ]),

    # Métriques
    html.Div(id="metrics-row", style={
        "display":"grid","gridTemplateColumns":"repeat(4,1fr)",
        "gap":"12px","padding":"16px 24px 0",
    }),

    # Contrôles fenêtre
    html.Div(style={"display":"flex","alignItems":"center","gap":"8px","padding":"12px 24px"},
    children=[
        html.Span("FENÊTRE :", style={"fontSize":"10px","color":MUTED}),
        html.Button("10s",  id="btn-10",  n_clicks=0,
                    style={"background":"transparent","border":f"1px solid {BDR}","color":MUTED,
                           "borderRadius":"6px","padding":"5px 12px","cursor":"pointer",
                           "fontFamily":"monospace","fontSize":"11px"}),
        html.Button("30s",  id="btn-30",  n_clicks=0,
                    style={"background":"transparent","border":f"1px solid {BLUE}","color":BLUE,
                           "borderRadius":"6px","padding":"5px 12px","cursor":"pointer",
                           "fontFamily":"monospace","fontSize":"11px"}),
        html.Button("60s",  id="btn-60",  n_clicks=0,
                    style={"background":"transparent","border":f"1px solid {BDR}","color":MUTED,
                           "borderRadius":"6px","padding":"5px 12px","cursor":"pointer",
                           "fontFamily":"monospace","fontSize":"11px"}),
        html.Button("Tout", id="btn-all", n_clicks=0,
                    style={"background":"transparent","border":f"1px solid {BDR}","color":MUTED,
                           "borderRadius":"6px","padding":"5px 12px","cursor":"pointer",
                           "fontFamily":"monospace","fontSize":"11px"}),
        dcc.Store(id="win-store", data=30),
    ]),

    # Banner alerte
    html.Div(id="alert-div", style={"padding":"0 24px 10px"}),

    # Graphiques
    html.Div(style={"padding":"0 24px","display":"flex","flexDirection":"column","gap":"12px"},
    children=[
        html.Div(style=CARD_STYLE, children=[
            html.Div("SIGNAL ECG BRUT — VOIE 1",
                     style={"fontSize":"10px","color":MUTED,
                            "letterSpacing":"2px","marginBottom":"8px"}),
            dcc.Graph(id="g-ecg", style={"height":"200px"},
                      config={"displayModeBar":False}),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px"},
        children=[
            html.Div(style=CARD_STYLE, children=[
                html.Div("PROBABILITÉ CRISE (SVM)",
                         style={"fontSize":"10px","color":MUTED,
                                "letterSpacing":"2px","marginBottom":"8px"}),
                dcc.Graph(id="g-prob", style={"height":"180px"},
                          config={"displayModeBar":False}),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.Div("HISTORIQUE ALERTES",
                         style={"fontSize":"10px","color":MUTED,
                                "letterSpacing":"2px","marginBottom":"8px"}),
                dcc.Graph(id="g-alert", style={"height":"180px"},
                          config={"displayModeBar":False}),
            ]),
        ]),
    ]),

    # Status bar
    html.Div(id="status-div", style={
        "padding":"10px 28px","borderTop":f"1px solid {BDR}","marginTop":"14px",
        "fontSize":"10px","color":MUTED,"display":"flex","gap":"24px",
    }),

    dcc.Interval(id="tick", interval=1000, n_intervals=0),
])


@app.callback(
    Output("win-store","data"),
    Input("btn-10","n_clicks"), Input("btn-30","n_clicks"),
    Input("btn-60","n_clicks"), Input("btn-all","n_clicks"),
    prevent_initial_call=True,
)
def set_win(b10, b30, b60, ball):
    ctx = callback_context
    if not ctx.triggered: return 30
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    return {"btn-10":10,"btn-30":30,"btn-60":60,"btn-all":99999}.get(btn, 30)


@app.callback(
    Output("metrics-row","children"),
    Output("g-ecg","figure"),
    Output("g-prob","figure"),
    Output("g-alert","figure"),
    Output("alert-div","children"),
    Output("status-div","children"),
    Input("tick","n_intervals"),
    Input("win-store","data"),
)
def update(_, win_s):
    with lock:
        t_raw  = list(hist_t)
        v_raw  = list(hist_ecg)
        t_prob = [x[0] for x in hist_prob]
        v_prob = [x[1] for x in hist_prob]
        t_al   = [x[0] for x in hist_al]
        v_al   = [x[1] for x in hist_al]
        n_buf  = len(buf_ch1)

    now_t = t_raw[-1] if t_raw else 0
    def trim(ts, vs):
        if not ts: return [],[]
        pairs = [(t,v) for t,v in zip(ts,vs) if t >= now_t - win_s]
        return ([p[0] for p in pairs],[p[1] for p in pairs]) if pairs else ([],[])

    t_ecg, v_ecg = trim(t_raw, v_raw)
    t_pr,  v_pr  = trim(t_prob, v_prob)
    t_al2, v_al2 = trim(t_al, v_al)

    buf_pct = int(n_buf / BUF_SIZE * 100)
    last_p  = f"{v_prob[-1]:.3f}" if v_prob else "—"
    uptime  = int(time.time() - start_time)
    upt_str = f"{uptime//3600:02d}:{(uptime%3600)//60:02d}:{uptime%60:02d}"
    mqtt_ok = mqtt_client.is_connected()

    def mcard(label, value, color, sub):
        return html.Div(style={
            **CARD_STYLE,"borderTop":f"2px solid {color}","padding":"12px 16px"
        }, children=[
            html.Div(label, style={"fontSize":"10px","color":MUTED,
                                   "letterSpacing":"1.5px","marginBottom":"6px"}),
            html.Div(value, style={"fontSize":"24px","fontWeight":"700","color":color}),
            html.Div(sub,   style={"fontSize":"11px","color":MUTED,"marginTop":"4px"}),
        ])

    metrics = [
        mcard("FRÉQUENCE",  f"{FS} Hz",   BLUE,  "Échantillonnage M5StickC"),
        mcard("BUFFER",     f"{buf_pct}%", GREEN, f"{n_buf:,} / {BUF_SIZE:,} samples"),
        mcard("PROBA SVM",  last_p,
              RED if v_prob and v_prob[-1]>=THR else BLUE,
              f"Seuil : {THR:.3f}"),
        mcard("ALERTES",    str(alert_count),
              RED if alert_count>0 else YEL,
              f"Durée : {upt_str}"),
    ]

    # ECG
    fig_ecg = go.Figure()
    if v_ecg:
        fig_ecg.add_trace(go.Scatter(
            x=t_ecg, y=v_ecg, mode="lines",
            line=dict(color=BLUE, width=1.2),
            fill="tozeroy",
            fillcolor="rgba(0,200,255,0.05)"))
    fig_ecg.update_layout(**PLOT, uirevision="ecg",
        yaxis_title="ADC", xaxis_title="Temps (s)")

    # Probabilité
    prob_layout = {**PLOT, "uirevision":"prob"}
    prob_layout["yaxis"] = dict(
        range=[-0.05,1.05], gridcolor=BDR, zerolinecolor=BDR)
    fig_prob = go.Figure()
    if v_pr:
        fig_prob.add_trace(go.Scatter(
            x=t_pr, y=v_pr, mode="lines+markers",
            line=dict(color=GREEN, width=2),
            marker=dict(size=4, color=GREEN),
            fill="tozeroy",
            fillcolor="rgba(0,255,156,0.05)"))
        fig_prob.add_hrect(y0=THR, y1=1.05,
                           fillcolor="rgba(255,69,96,0.07)", line_width=0)
    fig_prob.add_hline(y=THR, line_color=RED, line_dash="dot", line_width=1.5,
                       annotation_text=f"seuil {THR:.3f}",
                       annotation_font_color=RED, annotation_font_size=9)
    fig_prob.update_layout(**prob_layout)

    # Alertes
    al_layout = {**PLOT, "uirevision":"al"}
    al_layout["yaxis"] = dict(
        range=[-0.1,1.5], gridcolor=BDR, zerolinecolor=BDR)
    fig_al = go.Figure()
    if t_al2:
        colors = [RED if v==1 else MUTED for v in v_al2]
        fig_al.add_trace(go.Bar(
            x=t_al2, y=v_al2, marker_color=colors, width=20))
        for ta, va in zip(t_al2, v_al2):
            if va == 1:
                fig_al.add_vline(x=ta, line_color=RED,
                                 line_dash="dot", line_width=1,
                                 annotation_text="⚠",
                                 annotation_font_color=RED)
    fig_al.update_layout(**al_layout)

    # Banner alerte
    banner = []
    if alert_count > 0 and v_al and v_al[-1] == 1:
        banner = html.Div(style={
            "background":"rgba(255,69,96,0.1)",
            "border":f"1px solid {RED}",
            "borderRadius":"8px","padding":"12px 16px",
            "display":"flex","alignItems":"center","gap":"10px",
            "color":RED,"fontSize":"13px",
        }, children=[
            html.Span("⚠", style={"fontSize":"22px"}),
            html.Div([
                html.Div(
                    f"ALERTE — Crise épileptique prédite pour {PATIENT_NAME}",
                    style={"fontWeight":"700","marginBottom":"4px"}
                ),
                html.Div(
                    f"Email → {', '.join(EMAIL_TO)} | SMS → {', '.join(TWILIO_TO)}",
                    style={"fontSize":"11px","opacity":"0.8"}
                ),
            ]),
        ])

    # Status
    mqtt_status = "EMQX ✅" if mqtt_ok else "EMQX ❌ reconnexion..."
    status = [
        html.Span(f"PATIENT : {PATIENT_NAME}"),
        html.Span(f"BROKER : {mqtt_status}"),
        html.Span(f"SAMPLES : {len(v_ecg):,}"),
        html.Span(f"FENÊTRE : {win_s}s"),
        html.Span(f"UPTIME : {upt_str}"),
        html.Span(
            f"● BUFFER {'PRÊT' if n_buf>=BUF_SIZE else f'{buf_pct}%'}",
            style={"color": GREEN if n_buf>=BUF_SIZE else YEL}
        ),
    ]

    return metrics, fig_ecg, fig_prob, fig_al, banner, status


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Dashboard → http://127.0.0.1:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)