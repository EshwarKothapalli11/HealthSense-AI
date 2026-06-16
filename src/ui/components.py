"""
components.py — Reusable HTML component builders for HealthSense AI.

Light glassmorphism styled components matching the medical theme.
"""

import gradio as gr


def build_header_html() -> str:
    """Build the hero header HTML block."""
    return """
    <div class="app-hero">
      <div class="app-hero-pill">
        <span class="dot-live"></span>
        AI-Powered · Deep Learning · Multi-Modal Analysis
      </div>
      <h1>🏥 HealthSense AI</h1>
      <p>
        A professional deep learning system for diabetes risk prediction,
        heart disease assessment, and mental health detection —
        unified into a single intelligent health report.
      </p>
    </div>
    """


def build_metric_card_html(label: str, value: float, unit: str = "%", color: str = "#4a90d9") -> str:
    """Build a single glass metric card."""
    bar_width = min(int(value), 100)
    return f"""
    <div class="metric-glass">
        <div class="metric-num" style="background: linear-gradient(135deg, {color}, #6b6bd6);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             background-clip: text;">
            {value}{unit}
        </div>
        <div class="metric-lbl">{label}</div>
        <div class="metric-bar">
            <div class="metric-bar-fill"
                 style="width: {bar_width}%;
                        background: linear-gradient(90deg, {color}, #6b6bd6);">
            </div>
        </div>
    </div>
    """


def build_badge_html(tier: str, composite: float) -> str:
    """Build a risk tier badge."""
    tier_map = {
        'Low Risk':      ('badge-low',      '🟢'),
        'Moderate Risk': ('badge-moderate',  '🟡'),
        'High Risk':     ('badge-high',      '🟠'),
        'Critical':      ('badge-critical',  '🔴'),
    }
    css_class, icon = tier_map.get(tier, ('badge-low', '⚪'))
    return f"""
    <div style="text-align:center; margin: 20px 0;">
        <span class="badge {css_class}" style="font-size:15px; padding: 10px 28px;">
            {icon} {tier} — Composite Score: {composite}%
        </span>
    </div>
    """


def build_recommendation_html(recommendations: list[str]) -> str:
    """Build styled recommendation rows."""
    icons = ["💊", "🥗", "🏃", "🧘", "💤", "🩺", "❤️"]
    items = ""
    for i, rec in enumerate(recommendations):
        icon = icons[i % len(icons)]
        items += f"""
        <div class="rec-row">
            <span class="rec-icon">{icon}</span>
            <span>{rec}</span>
        </div>
        """
    return f"""
    <div class="glass" style="padding: 24px;">
        <p class="sec-title">💡 Personalized Recommendations</p>
        {items}
    </div>
    """


def build_mental_status_html(status: str, confidence: float) -> str:
    """Build a mental health status display card."""
    if status == "Depressed/Stressed":
        color, icon, badge = "#f56565", "🧠", "badge-high"
    else:
        color, icon, badge = "#48bb78", "✅", "badge-low"
    return f"""
    <div class="metric-glass" style="text-align:center;">
        <div style="font-size: 48px; margin-bottom:12px;">{icon}</div>
        <div style="font-size: 22px; font-weight: 700; color: {color}; margin-bottom:8px;">
            {status}
        </div>
        <span class="badge {badge}">Confidence: {confidence}%</span>
        <div class="metric-bar" style="margin-top:16px;">
            <div class="metric-bar-fill"
                 style="width:{int(confidence)}%;
                        background: linear-gradient(90deg, {color}, {color}88);">
            </div>
        </div>
    </div>
    """


def build_about_html() -> str:
    """Build the About & Architecture section."""
    return """
    <div class="glass" style="padding: 32px;">
        <h2 style="color:#1a3a7a;">HealthSense AI — System Overview</h2>
        <hr class="glass-hr">
        <p class="sec-title">🧬 Architecture</p>
        <p style="color:#2d4060;">Three independent deep learning models trained on different data modalities,
        fused through a weighted scoring engine into a single unified health report.</p>
        <hr class="glass-hr">
        <p class="sec-title">🛠️ Tech Stack</p>
        <p style="color:#2d4060;">TensorFlow / Keras &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp;
           Gradio 4.x &nbsp;·&nbsp; Python 3.10+ &nbsp;·&nbsp; Plotly &nbsp;·&nbsp;
           matplotlib + seaborn</p>
        <hr class="glass-hr">
        <p class="sec-title">🤖 Models</p>
        <p style="color:#2d4060;"><strong>Diabetes XGBoost</strong> — Gradient-boosted tree ensemble (150 estimators, depth 3)</p>
        <p style="color:#2d4060;"><strong>Heart Disease XGBoost</strong> — Gradient-boosted tree ensemble (100 estimators, depth 4)</p>
        <p style="color:#2d4060;"><strong>Mental Health LSTM</strong> — Embedding(64) → LSTM(64) → Dense(32)</p>
        <hr class="glass-hr">
        <p class="sec-title">🔗 Links</p>
        <p style="color:#2d4060;">GitHub: <a href="#" style="color:#4a72b8;">github.com/healthsense-ai</a></p>
        <p style="color:#2d4060;">LinkedIn: <a href="#" style="color:#4a72b8;">linkedin.com/healthsense</a></p>
    </div>
    """


def build_diabetes_inputs(prefix: str = "") -> list:
    """Create diabetes risk factor input sliders."""
    return [
        gr.Slider(0,  17,   value=1,    step=1,    label="Pregnancies"),
        gr.Slider(0,  200,  value=110,  step=1,    label="Glucose"),
        gr.Slider(0,  140,  value=72,   step=1,    label="Blood Pressure"),
        gr.Slider(0,  100,  value=20,   step=1,    label="Skin Thickness"),
        gr.Slider(0,  900,  value=80,   step=1,    label="Insulin"),
        gr.Slider(10, 70,   value=25.0, step=0.1,  label="BMI"),
        gr.Slider(0,  2.5,  value=0.47, step=0.01, label="Diabetes Pedigree Function"),
        gr.Slider(1,  120,  value=30,   step=1,    label="Age"),
    ]


def build_heart_inputs(prefix: str = "") -> list:
    """Create heart disease risk factor inputs."""
    return [
        gr.Slider(1,   120, value=50,  step=1,   label="Age"),
        gr.Radio(["Male", "Female"],              label="Sex", value="Male"),
        gr.Dropdown(["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"], label="Chest Pain Type", value="Typical Angina"),
        gr.Slider(80,  200, value=120, step=1,   label="Resting Blood Pressure"),
        gr.Slider(100, 400, value=200, step=1,   label="Cholesterol"),
        gr.Checkbox(label="Fasting Blood Sugar > 120 mg/dl", value=False),
        gr.Dropdown(["Normal", "ST-T Abnormality", "LV Hypertrophy"], label="Resting ECG", value="Normal"),
        gr.Slider(60,  220, value=150, step=1,   label="Max Heart Rate Achieved"),
        gr.Checkbox(label="Exercise Induced Angina",          value=False),
        gr.Slider(0,   7,   value=1.0, step=0.1, label="ST Depression (Oldpeak)"),
        gr.Dropdown(["Up", "Flat", "Down"],       label="ST Slope", value="Flat"),
        gr.Slider(0,   3,   value=0,   step=1,   label="Number of Major Vessels (CA)"),
        gr.Dropdown(["Normal", "Fixed Defect", "Reversable Defect"], label="Thalassemia", value="Normal"),
    ]
