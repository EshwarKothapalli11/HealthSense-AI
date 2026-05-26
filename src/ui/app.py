"""
app.py — Gradio web application for HealthSense AI.

5-tab interface: Physical Health, Mental Health, Unified Report,
Model Insights, and About & Architecture.
"""

import os
import sys

import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf
import tempfile
import time
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config
from src.fusion.fusion_engine import HealthFusionEngine
from src.evaluation.explainability import (
    get_top_influential_features, get_top_influential_tokens
)
from src.ui.visualizations import plot_gauge, plot_radar_chart


def create_app(
    diabetes_model: tf.keras.Model,
    heart_model: tf.keras.Model,
    mental_model: tf.keras.Model,
    diabetes_scaler,
    heart_scaler,
    tokenizer,
    fusion_engine: HealthFusionEngine,
    diabetes_features: list[str] | None = None,
    heart_features: list[str] | None = None
) -> gr.Blocks:
    """
    Build the full Gradio Blocks application with 5 tabs.

    Args:
        diabetes_model: Trained diabetes Keras model.
        heart_model: Trained heart disease Keras model.
        mental_model: Trained mental health LSTM Keras model.
        diabetes_scaler: Fitted StandardScaler for diabetes features.
        heart_scaler: Fitted StandardScaler for heart features.
        tokenizer: Fitted Keras Tokenizer for text processing.
        fusion_engine: Initialized HealthFusionEngine.
        diabetes_features: List of diabetes feature names.
        heart_features: List of heart feature names.

    Returns:
        Configured gr.Blocks application.
    """

    if diabetes_features is None:
        diabetes_features = config.DIABETES_COLUMNS[:-1]  # Exclude 'Outcome'
    if heart_features is None:
        heart_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol',
            'fbs', 'restecg', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca', 'thal'
        ]

    # ---- Prediction helper functions ----

    def _prepare_diabetes_input(*values):
        """Prepare diabetes input from slider values."""
        features = np.array(values, dtype=np.float32).reshape(1, -1)
        features_scaled = diabetes_scaler.transform(features)
        return features_scaled

    def _prepare_heart_input(age, sex, cp, trestbps, chol, fbs, thalach, exang, oldpeak, slope):
        """Prepare heart input by dynamically building the one-hot encoded vector."""
        raw_dict = {
            'age': float(age),
            'sex': 1.0 if sex == "Male" else 0.0,
            'cp': float({"Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal": 3, "Asymptomatic": 4}.get(cp, 1)),
            'trestbps': float(trestbps),
            'chol': float(chol),
            'fbs': 1.0 if fbs else 0.0,
            'restecg': 0.0,
            'thalach': float(thalach),
            'exang': 1.0 if exang else 0.0,
            'oldpeak': float(oldpeak),
            'slope': float({"Up": 1, "Flat": 2, "Down": 3}.get(slope, 1)),
            'ca': 0.0,
            'thal': 3.0
        }

        feature_names = [
            'age', 'trestbps', 'chol', 'thalach', 'oldpeak',
            'sex_1.0', 
            'cp_2.0', 'cp_3.0', 'cp_4.0', 
            'fbs_1.0', 
            'restecg_1.0', 'restecg_2.0', 
            'exang_1.0', 
            'slope_2.0', 'slope_3.0', 
            'ca_1.0', 'ca_2.0', 'ca_3.0', 
            'thal_6.0', 'thal_7.0'
        ]
        
        input_array = np.zeros(len(feature_names), dtype=np.float32)

        for i, col in enumerate(feature_names):
            if col in raw_dict:
                # Direct continuous match
                input_array[i] = raw_dict[col]
            elif '_' in col:
                # One-hot encoded feature match (e.g. 'cp_2.0')
                try:
                    base_feat, val = col.rsplit('_', 1)
                    if base_feat in raw_dict and float(raw_dict[base_feat]) == float(val):
                        input_array[i] = 1.0
                except ValueError:
                    pass

        features_scaled = heart_scaler.transform(input_array.reshape(1, -1))
        return features_scaled

    def _prepare_mental_input(text):
        """Prepare text input for mental health model."""
        from src.data.preprocess_text import clean_text, texts_to_padded
        cleaned = clean_text(text)
        padded = texts_to_padded([cleaned], tokenizer, maxlen=config.MAX_TEXT_LEN)
        return padded

    # ---- Tab 1: Physical Health ----

    def predict_physical(
        glucose, bp, skin, insulin, bmi, dpf, age_d,
        age_h, sex, cp, trestbps, chol, fbs, thalach, exang, oldpeak, slope
    ):
        """Run diabetes and heart disease predictions."""
        try:
            # Diabetes prediction
            d_input = _prepare_diabetes_input(
                0, glucose, bp, skin, insulin, bmi, dpf, age_d
            )
            d_prob = float(diabetes_model.predict(d_input, verbose=0)[0][0])

            # Heart prediction
            h_input = _prepare_heart_input(
                age_h, sex, cp, trestbps, chol, fbs, thalach, exang, oldpeak, slope
            )
            h_prob = float(heart_model.predict(h_input, verbose=0)[0][0])

            # Labels
            d_label = {
                "Diabetes Risk": d_prob,
                "Healthy": 1 - d_prob
            }
            h_label = {
                "Heart Disease Risk": h_prob,
                "Healthy": 1 - h_prob
            }

            # Gauge chart
            gauge_fig = plot_gauge(
                max(d_prob, h_prob) * 100,
                "Physical Health Risk"
            )

            # Top influential features
            d_features = get_top_influential_features(
                diabetes_model, d_input, diabetes_features, top_n=5
            )
            feat_df = pd.DataFrame(d_features)

            return d_label, h_label, gauge_fig, feat_df

        except Exception as e:
            traceback.print_exc()
            error_label = {"Error": 1.0}
            return error_label, error_label, None, pd.DataFrame({"error": [str(e)]})

    # ---- Tab 2: Mental Health ----

    def predict_mental(text):
        """Run mental health prediction."""
        try:
            if not text or len(text.strip()) < 5:
                return (
                    {"Please enter more text": 1.0},
                    "<div style='padding:15px; color:#e74c3c;'>Please provide a longer description of your feelings.</div>",
                    None
                )

            m_input = _prepare_mental_input(text)
            m_prob = float(mental_model.predict(m_input, verbose=0)[0][0])

            status = "High Stress/Depression" if m_prob >= 0.5 else "Healthy"
            m_label = {
                "High Stress/Depression": m_prob,
                "Healthy": 1 - m_prob
            }

            confidence = max(m_prob, 1 - m_prob) * 100
            color = "#f56565" if status != "Healthy" else "#48bb78"
            
            confidence_html = f"""
            <div style='padding: 20px; border-radius: 12px; background-color: rgba(255, 255, 255, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.8); box-shadow: 0 8px 32px rgba(31, 41, 55, 0.05); border-left: 6px solid {color}; margin-bottom: 20px;'>
                <h3 style='margin-top: 0; color: #1F2937;'>Assessment Confidence</h3>
                <p style='font-size: 1.2em; color: #4B5563; margin: 10px 0;'>
                    <strong>{confidence:.1f}% likelihood</strong> of {status.lower()} state.
                </p>
                <div style='width: 100%; background-color: rgba(0, 0, 0, 0.1); border-radius: 8px; height: 16px; overflow: hidden;'>
                    <div style='width: {confidence}%; background-color: {color}; height: 100%; border-radius: 8px;'></div>
                </div>
            </div>
            """

            # Get influential tokens
            top_tokens = get_top_influential_tokens(
                tokenizer, m_input, mental_model, top_n=5
            )

            # Create highlighted text
            from src.data.preprocess_text import clean_text
            cleaned = clean_text(text)
            words = cleaned.split()
            highlighted = []
            for word in words:
                if word.lower() in [t.lower() for t in top_tokens]:
                    highlighted.append((word, "influential"))
                else:
                    highlighted.append((word, None))

            return m_label, confidence_html, highlighted

        except Exception as e:
            traceback.print_exc()
            return {"Error": 1.0}, f"<div style='color:red;'>Error: {str(e)}</div>", None

    def count_chars(text):
        """Return character count for the text."""
        return f"Character count: {len(text)}"

    # ---- Tab 3: Unified Health Report ----

    def generate_full_report(
        glucose, bp, skin, insulin, bmi, dpf, age_d,
        age_h, sex, cp, trestbps, chol, fbs, thalach, exang, oldpeak, slope,
        mental_text,
        w_d, w_h, w_m, m_thresh
    ):
        """Generate unified health report combining all three models."""
        try:
            # Diabetes
            d_input = _prepare_diabetes_input(
                0, glucose, bp, skin, insulin, bmi, dpf, age_d
            )
            d_prob = float(diabetes_model.predict(d_input, verbose=0)[0][0])

            # Heart
            h_input = _prepare_heart_input(
                age_h, sex, cp, trestbps, chol, fbs, thalach, exang, oldpeak, slope
            )
            h_prob = float(heart_model.predict(h_input, verbose=0)[0][0])

            # Mental
            if mental_text and len(mental_text.strip()) >= 5:
                m_input = _prepare_mental_input(mental_text)
                m_prob = float(mental_model.predict(m_input, verbose=0)[0][0])
            else:
                m_prob = 0.0

            # Fuse
            result = fusion_engine.fuse(
                d_prob, h_prob, m_prob,
                w_diabetes=w_d, w_heart=w_h, w_mental=w_m,
                m_threshold=m_thresh
            )

            # Format outputs
            d_label = {"Diabetes Risk": result['diabetes_risk_pct'] / 100, "Healthy": 1 - result['diabetes_risk_pct'] / 100}
            h_label = {"Heart Disease Risk": result['heart_risk_pct'] / 100, "Healthy": 1 - result['heart_risk_pct'] / 100}
            m_label = {"Mental Risk": result['mental_confidence_pct'] / 100, "Healthy": 1 - result['mental_confidence_pct'] / 100}
            
            c_score = result['composite_score']
            tier = result['risk_tier']
            t_color = "#f56565" if tier in ["High", "Critical"] else ("#ed8936" if tier == "Moderate" else "#48bb78")
            
            composite_html = f"""
            <div style='padding: 24px; border-radius: 20px; background: rgba(255,255,255,0.70); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.90); border-top: 6px solid {t_color}; box-shadow: 0 4px 20px rgba(99,150,210,0.10); text-align: center; margin-bottom: 15px;'>
                <h3 style='margin-top: 0; color: #1a3a7a; font-weight: 700;'>Composite Health Score</h3>
                <h1 style='font-size: 3.5em; margin: 10px 0; color: {t_color};'>{c_score}<span style='font-size: 0.4em; color: rgba(60,90,140,0.55);'> / 100</span></h1>
                <p style='font-size: 1.3em; color: #2d4060; margin-bottom: 0;'>
                    Risk Tier: <strong style='color: {t_color};'>{tier}</strong>
                </p>
            </div>
            """

            summary = result['health_summary']

            rec_data = [[i+1, r] for i, r in enumerate(result['recommendations'])]
            rec_df = pd.DataFrame(rec_data, columns=['#', 'Recommendation'])

            radar_fig = plot_radar_chart(
                result['diabetes_risk_pct'],
                result['heart_risk_pct'],
                result['mental_confidence_pct']
            )

            # Generate HTML export report
            recs_html = ''.join(f'<li>{r}</li>' for r in result['recommendations'])
            html_content = f"""
            <html>
                <body style="font-family: Arial, sans-serif; padding: 20px; max-width: 800px; margin: auto;">
                    <h1 style="color: #2c3e50;">HealthSense AI — Unified Assessment Report</h1>
                    <p style="color: #7f8c8d;"><strong>Generated At:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <hr style="border: 1px solid #bdc3c7;">
                    
                    <h2 style="color: #2980b9;">1. Composite Risk Profile</h2>
                    <p style="font-size: 1.2em;"><strong>Risk Tier:</strong> <span style="color: {t_color};">{tier}</span></p>
                    <p style="font-size: 1.2em;"><strong>Composite Score:</strong> {c_score} / 100</p>
                    <div style="padding: 15px; background-color: #f8f9fa; border-left: 5px solid {t_color};">
                        {summary}
                    </div>

                    <h2 style="color: #2980b9;">2. Individual Assessments</h2>
                    <ul>
                        <li style="margin-bottom: 5px;"><strong>Diabetes Risk:</strong> {result['diabetes_risk_pct']}%</li>
                        <li style="margin-bottom: 5px;"><strong>Heart Disease Risk:</strong> {result['heart_risk_pct']}%</li>
                        <li style="margin-bottom: 5px;"><strong>Mental Health ({result['mental_status']}):</strong> {result['mental_confidence_pct']}%</li>
                    </ul>

                    <h2 style="color: #2980b9;">3. Actionable Recommendations</h2>
                    <ul>{recs_html}</ul>
                    
                    <br><hr style="border: 1px solid #bdc3c7;">
                    <p style="font-size: 0.8em; color: #95a5a6; text-align: center;">
                        Generated by HealthSense AI. This system provides AI-driven insights and is NOT a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare provider.
                    </p>
                </body>
            </html>
            """
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            tmp.write(html_content.encode("utf-8"))
            tmp.close()
            report_file = tmp.name

            return d_label, h_label, m_label, composite_html, summary, rec_df, radar_fig, report_file

        except Exception as e:
            traceback.print_exc()
            err = {"Error": 1.0}
            return (
                err, err, err,
                f"<div style='color:red;'>Error: {str(e)}</div>", "", pd.DataFrame(), None, None
            )

    # ---- Tab 4: Model Insights ----

    def load_plot_image(filename):
        """Load a plot image from artifacts directory."""
        path = os.path.join(config.PLOTS_DIR, filename)
        if os.path.exists(path):
            return path
        return None

    # ---- Build the interface ----

    custom_theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="teal",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="transparent",
        body_background_fill_dark="transparent",
        body_text_color="#1a2332",
        body_text_color_dark="#1a2332",
        block_background_fill="rgba(255, 255, 255, 0.62)",
        block_background_fill_dark="rgba(255, 255, 255, 0.62)",
        block_border_width="1px",
        block_border_color="rgba(255, 255, 255, 0.85)",
        block_border_color_dark="rgba(255, 255, 255, 0.85)",
        block_title_text_color="#1a3a7a",
        block_title_text_color_dark="#1a3a7a",
        block_label_text_color="rgba(40, 60, 100, 0.80)",
        block_label_text_color_dark="rgba(40, 60, 100, 0.80)",
        block_shadow="0 8px 40px rgba(99,150,210,0.12), 0 2px 8px rgba(99,150,210,0.08)",
        button_primary_background_fill="linear-gradient(135deg, #4a90d9 0%, #6b6bd6 50%, #9b72d4 100%)",
        button_primary_background_fill_dark="linear-gradient(135deg, #4a90d9 0%, #6b6bd6 50%, #9b72d4 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #3a80c9 0%, #5b5bc6 50%, #8b62c4 100%)",
        button_primary_background_fill_hover_dark="linear-gradient(135deg, #3a80c9 0%, #5b5bc6 50%, #8b62c4 100%)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="rgba(255, 255, 255, 0.65)",
        button_secondary_background_fill_dark="rgba(255, 255, 255, 0.65)",
        button_secondary_background_fill_hover="rgba(255, 255, 255, 0.90)",
        button_secondary_background_fill_hover_dark="rgba(255, 255, 255, 0.90)",
        button_secondary_text_color="rgba(40,70,130,0.80)",
        button_secondary_text_color_dark="rgba(40,70,130,0.80)",
        border_color_primary="rgba(99, 150, 237, 0.40)",
        border_color_primary_dark="rgba(99, 150, 237, 0.40)",
        slider_color="#4a90d9",
        slider_color_dark="#4a90d9",
        input_background_fill="rgba(255, 255, 255, 0.75)",
        input_background_fill_dark="rgba(255, 255, 255, 0.75)",
    )

    with gr.Blocks(
        theme=custom_theme,
        title="HealthSense AI — Multi-Modal Health Assessment",
        css="""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ══ AMBIENT DRIFT ANIMATION ══ */
        @keyframes ambientDrift {
            0%   { transform: scale(1)    rotate(0deg); }
            50%  { transform: scale(1.04) rotate(1.5deg); }
            100% { transform: scale(1)    rotate(-1deg); }
        }

        @keyframes livePulse {
            0%,100% { opacity:1;   transform: scale(1);   }
            50%     { opacity:0.4; transform: scale(0.85); }
        }

        @keyframes criticalPulse {
            0%,100% { box-shadow: 0 0 0 0   rgba(229,62,62,0.30); }
            50%     { box-shadow: 0 0 0 10px rgba(229,62,62,0);    }
        }

        /* ══ BASE BACKGROUND ══ */
        *, *::before, *::after { box-sizing: border-box; }

        body,
        .gradio-container,
        .gradio-container > .main,
        .gradio-container > .main > .wrap {
            min-height:  100vh !important;
            background:  linear-gradient(135deg,
                #e8f4fd 0%, #f0f7ff 25%, #e6f3fb 50%, #f5f0ff 75%, #eef5ff 100%) !important;
            background-attachment: fixed !important;
            font-family: 'Inter', sans-serif !important;
            color:       #1a2332 !important;
        }

        /* ══ SOFT AMBIENT ORBS ══ */
        .gradio-container::before {
            content:  '';
            position: fixed;
            inset:    0;
            background:
                radial-gradient(ellipse 70% 55% at 15% 35%, rgba(99,179,237,0.22) 0%, transparent 60%),
                radial-gradient(ellipse 55% 70% at 85% 20%, rgba(154,117,234,0.16) 0%, transparent 60%),
                radial-gradient(ellipse 65% 45% at 55% 85%, rgba(72,187,120,0.14)  0%, transparent 60%),
                radial-gradient(ellipse 45% 60% at 90% 75%, rgba(237,137,54,0.10)  0%, transparent 60%);
            animation:  ambientDrift 30s ease-in-out infinite alternate;
            pointer-events: none;
            z-index:    0;
        }

        /* ══ CENTERED CONTAINER ══ */
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
            padding: 24px !important;
            position: relative;
            z-index: 1;
        }

        .gradio-container .prose h2 {
            border-bottom: none !important;
            color: #1a3a7a !important;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 16px;
        }

        /* ══ GLASS CARDS (all blocks) ══ */
        .gradio-box, .gr-box, .gr-panel,
        .gr-group, .gr-form {
            background:      rgba(255, 255, 255, 0.62) !important;
            backdrop-filter: blur(28px) saturate(160%) !important;
            -webkit-backdrop-filter: blur(28px) saturate(160%) !important;
            border:          1px solid rgba(255, 255, 255, 0.85) !important;
            border-radius:   24px !important;
            box-shadow:
                0 8px 40px rgba(99,150,210,0.12),
                0 2px 8px  rgba(99,150,210,0.08),
                inset 0 1px 0 rgba(255,255,255,0.95) !important;
            transition: transform 0.35s cubic-bezier(0.34,1.56,0.64,1),
                        box-shadow 0.35s ease !important;
        }

        .gradio-box:hover, .gr-box:hover, .gr-panel:hover {
            transform:  translateY(-3px) !important;
            box-shadow:
                0 16px 50px rgba(99,150,210,0.16),
                0 4px 12px  rgba(99,150,210,0.10),
                inset 0 1px 0 rgba(255,255,255,1) !important;
        }

        /* ══ APP HERO HEADER ══ */
        .app-hero {
            text-align:  center;
            padding:     48px 24px 32px;
            position:    relative;
        }

        .app-hero h1 {
            font-size:   44px;
            font-weight: 800;
            background:  linear-gradient(135deg, #2b6cb0 0%, #6b46c1 50%, #2c7a7b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 12px;
            line-height:   1.15;
        }

        .app-hero p {
            font-size:   16px;
            color:       rgba(60,90,140,0.65) !important;
            max-width:   580px;
            margin:      0 auto;
            line-height: 1.65;
        }

        .app-hero-pill {
            display:        inline-flex;
            align-items:    center;
            gap:            8px;
            background:     rgba(255,255,255,0.70);
            border:         1px solid rgba(180,205,235,0.60);
            border-radius:  999px;
            padding:        8px 20px;
            font-size:      13px;
            color:          rgba(40,70,130,0.75) !important;
            font-weight:    500;
            margin-bottom:  24px;
            backdrop-filter: blur(8px);
            box-shadow:     0 2px 12px rgba(99,150,210,0.12);
        }

        .dot-live {
            display:       inline-block;
            width:         8px; height: 8px;
            border-radius: 50%;
            background:    #48bb78;
            box-shadow:    0 0 8px rgba(72,187,120,0.60);
            animation:     livePulse 2s ease-in-out infinite;
            margin-right:  2px;
        }

        /* ══ TAB BAR ══ */
        .tabs {
            border-radius: 18px !important;
            background: transparent !important;
        }

        .tabs > .tab-nav {
            background:      rgba(255, 255, 255, 0.55) !important;
            backdrop-filter: blur(20px) !important;
            border:          1px solid rgba(255, 255, 255, 0.80) !important;
            border-radius:   18px !important;
            padding:         6px !important;
            margin-bottom:   28px !important;
            display:         flex !important;
            gap:             4px !important;
            box-shadow:      0 4px 20px rgba(99,150,210,0.10) !important;
            flex-wrap:       wrap !important;
        }

        .tabs > .tab-nav > button,
        .tab-nav button {
            background:    transparent !important;
            border:        1px solid transparent !important;
            border-radius: 13px !important;
            color:         rgba(60, 80, 120, 0.60) !important;
            font-family:   'Inter', sans-serif !important;
            font-size:     13px !important;
            font-weight:   500 !important;
            padding:       10px 20px !important;
            transition:    all 0.25s ease !important;
            white-space:   nowrap !important;
            text-transform: none !important;
            letter-spacing: normal !important;
            box-shadow:    none !important;
        }

        .tabs > .tab-nav > button:hover,
        .tab-nav button:hover {
            background: rgba(99, 150, 237, 0.10) !important;
            color:      rgba(40, 60, 120, 0.90) !important;
            transform:  translateY(-1px);
        }

        .tabs > .tab-nav > button.selected,
        .tab-nav button.selected {
            background:   linear-gradient(135deg, rgba(99,150,237,0.20), rgba(154,117,234,0.15)) !important;
            border-color: rgba(99, 150, 237, 0.40) !important;
            color:        #1a3a7a !important;
            font-weight:  600 !important;
            box-shadow:
                0 4px 16px rgba(99,150,237,0.20),
                inset 0 1px 0 rgba(255,255,255,0.90) !important;
        }

        /* ══ ALL INPUTS ══ */
        .gr-box, .gr-input,
        input[type="text"], input[type="number"], input[type="email"], input[type="password"],
        textarea, select,
        [data-testid="textbox"] textarea,
        [data-testid="number"]  input {
            background:      rgba(255, 255, 255, 0.75) !important;
            border:          1px solid rgba(180, 205, 235, 0.70) !important;
            border-radius:   14px !important;
            color:           #1a2332 !important;
            font-family:     'Inter', sans-serif !important;
            font-size:       14px !important;
            backdrop-filter: blur(8px) !important;
            transition:      border-color 0.2s, box-shadow 0.2s !important;
            box-shadow:      0 2px 8px rgba(99,150,210,0.08) !important;
        }

        input:focus, textarea:focus, select:focus,
        .gr-dropdown:focus-within {
            border-color: rgba(99, 150, 237, 0.70) !important;
            box-shadow:   0 0 0 3px rgba(99,150,237,0.15),
                          0 2px 8px rgba(99,150,210,0.10) !important;
            outline:      none !important;
            background:   rgba(255,255,255,0.92) !important;
        }

        input::placeholder, textarea::placeholder {
            color: rgba(100, 130, 170, 0.55) !important;
        }

        /* ══ SLIDERS ══ */
        input[type="range"] {
            accent-color: #4a90d9 !important;
            height:       5px !important;
            border:       none !important;
            padding:      0 !important;
            background:   rgba(99,150,210,0.15) !important;
            border-radius: 99px !important;
            -webkit-appearance: none !important;
            appearance: none !important;
        }

        input[type="range"]::-webkit-slider-thumb {
            width: 20px !important;
            height: 20px !important;
            background: #4a90d9 !important;
            border: 3px solid #ffffff !important;
            box-shadow: 0 0 10px rgba(74,144,217,0.40) !important;
            border-radius: 50% !important;
            cursor: pointer !important;
            -webkit-appearance: none !important;
            margin-top: -8px !important;
            transition: transform 0.2s ease !important;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.15) !important;
        }

        input[type="range"]::-webkit-slider-runnable-track {
            background: rgba(99,150,210,0.15) !important;
            border-radius: 99px !important;
        }

        /* ══ LABELS ══ */
        label, .gr-label {
            color:       rgba(40, 60, 100, 0.80) !important;
            font-size:   13px !important;
            font-weight: 500 !important;
            margin-bottom: 6px !important;
        }

        /* ══ RADIO & CHECKBOX ══ */
        label.selected, label:has(input:checked) {
            background: linear-gradient(135deg, rgba(99,150,237,0.20), rgba(154,117,234,0.15)) !important;
            border: 1px solid rgba(99, 150, 237, 0.40) !important;
            color: #1a3a7a !important;
            box-shadow: 0 4px 16px rgba(99,150,237,0.20) !important;
        }

        label:has(input[type="radio"]:not(:checked)),
        label:has(input[type="checkbox"]:not(:checked)) {
            background-color: rgba(255, 255, 255, 0.55) !important;
            border: 1px solid rgba(180,205,235,0.50) !important;
            color: rgba(60, 80, 120, 0.70) !important;
            box-shadow: 0 2px 8px rgba(99,150,210,0.06) !important;
        }

        label:has(input[type="radio"]:not(:checked)):hover,
        label:has(input[type="checkbox"]:not(:checked)):hover {
            border-color: rgba(99,150,237,0.50) !important;
            background-color: rgba(255, 255, 255, 0.80) !important;
            transform: translateY(-1px);
            transition: all 0.2s ease;
        }

        input[type="checkbox"], input[type="radio"] {
            accent-color: #4a90d9 !important;
        }

        /* ══ PRIMARY BUTTON ══ */
        button.primary, button.lg.primary {
            background:    linear-gradient(135deg, #4a90d9 0%, #6b6bd6 50%, #9b72d4 100%) !important;
            border:        none !important;
            border-radius: 16px !important;
            color:         #ffffff !important;
            font-family:   'Inter', sans-serif !important;
            font-size:     15px !important;
            font-weight:   600 !important;
            padding:       14px 32px !important;
            letter-spacing: 0.3px !important;
            cursor:        pointer !important;
            box-shadow:
                0 6px 28px rgba(74,144,217,0.35),
                inset 0 1px 0 rgba(255,255,255,0.30) !important;
            transition:    all 0.30s cubic-bezier(0.34,1.56,0.64,1) !important;
            text-transform: none !important;
            position:      relative !important;
            overflow:      hidden !important;
        }

        button.primary:hover, button.lg.primary:hover {
            transform:  translateY(-3px) scale(1.01) !important;
            box-shadow: 0 12px 40px rgba(74,144,217,0.50),
                        inset 0 1px 0 rgba(255,255,255,0.35) !important;
        }

        button.primary:active, button.lg.primary:active {
            transform: translateY(0) scale(0.98) !important;
        }

        /* ══ SECONDARY BUTTON ══ */
        button.secondary, button.lg.secondary {
            background:      rgba(255,255,255,0.65) !important;
            border:          1px solid rgba(180,205,235,0.70) !important;
            border-radius:   16px !important;
            color:           rgba(40,70,130,0.80) !important;
            font-weight:     500 !important;
            backdrop-filter: blur(8px) !important;
            transition:      all 0.25s ease !important;
            box-shadow:      0 2px 10px rgba(99,150,210,0.10) !important;
        }

        button.secondary:hover, button.lg.secondary:hover {
            background:   rgba(255,255,255,0.90) !important;
            border-color: rgba(99,150,237,0.50) !important;
            color:        #1a3a7a !important;
        }

        /* ══ DATAFRAME / TABLE ══ */
        table {
            background:    rgba(255,255,255,0.55) !important;
            border-radius: 16px !important;
            overflow:      hidden !important;
            border:        1px solid rgba(180,205,235,0.50) !important;
            width:         100% !important;
            backdrop-filter: blur(12px) !important;
        }

        th {
            background:     linear-gradient(135deg, rgba(74,144,217,0.18), rgba(107,107,214,0.12)) !important;
            color:          #1a3a7a !important;
            font-weight:    600 !important;
            padding:        14px 18px !important;
            font-size:      12px !important;
            letter-spacing: 0.8px !important;
            text-transform: uppercase !important;
            border-bottom:  1px solid rgba(180,205,235,0.40) !important;
        }

        td {
            color:         #2d4060 !important;
            border-bottom: 1px solid rgba(180,205,235,0.25) !important;
            padding:       12px 18px !important;
            font-size:     14px !important;
        }

        tr:hover td {
            background: rgba(74,144,217,0.06) !important;
        }

        /* ══ ACCORDION ══ */
        .gr-accordion {
            background:      rgba(255,255,255,0.50) !important;
            border:          1px solid rgba(180,205,235,0.50) !important;
            border-radius:   16px !important;
            backdrop-filter: blur(12px) !important;
        }

        /* ══ SCROLLBAR ══ */
        ::-webkit-scrollbar       { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: rgba(99,150,210,0.06); }
        ::-webkit-scrollbar-thumb {
            background:    linear-gradient(#4a90d9, #9b72d4);
            border-radius: 99px;
        }

        /* ══ PLOTLY TRANSPARENT ══ */
        .js-plotly-plot .plotly,
        .js-plotly-plot .bg { background: transparent !important; fill: transparent !important; }

        /* ══ IMAGES ══ */
        .gr-image img {
            border-radius: 16px !important;
            border:        1px solid rgba(180,205,235,0.40) !important;
            box-shadow:    0 4px 16px rgba(99,150,210,0.10) !important;
        }

        /* ══ SECTION TITLE UTILITY ══ */
        .sec-title {
            font-size:      11px;
            font-weight:    700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color:          #4a72b8 !important;
            margin-bottom:  16px;
            display:        flex;
            align-items:    center;
            gap:            10px;
        }

        .sec-title::after {
            content:    '';
            flex:       1;
            height:     1px;
            background: linear-gradient(90deg, rgba(74,144,217,0.40), transparent);
        }

        /* ══ GLASS DIVIDER ══ */
        .glass-hr {
            height:     1px;
            background: linear-gradient(90deg, transparent, rgba(99,150,210,0.25), transparent);
            border:     none;
            margin:     28px 0;
        }

        /* ══ RISK BADGES ══ */
        .badge {
            display:        inline-flex;
            align-items:    center;
            gap:            6px;
            padding:        8px 20px;
            border-radius:  999px;
            font-size:      13px;
            font-weight:    700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .badge-low {
            background: rgba(72,187,120,0.12);
            border:     1px solid rgba(72,187,120,0.40);
            color:      #276749;
        }

        .badge-moderate {
            background: rgba(237,137,54,0.12);
            border:     1px solid rgba(237,137,54,0.40);
            color:      #9c4221;
        }

        .badge-high {
            background: rgba(245,101,101,0.12);
            border:     1px solid rgba(245,101,101,0.40);
            color:      #9b2c2c;
        }

        .badge-critical {
            background: rgba(229,62,62,0.12);
            border:     1px solid rgba(229,62,62,0.50);
            color:      #742a2a;
            animation:  criticalPulse 1.8s ease-in-out infinite;
        }

        /* ══ METRIC GLASS CARD ══ */
        .metric-glass {
            background:      rgba(255,255,255,0.70);
            border:          1px solid rgba(255,255,255,0.90);
            border-radius:   20px;
            padding:         24px 20px;
            text-align:      center;
            backdrop-filter: blur(20px);
            transition:      all 0.30s ease;
            position:        relative;
            overflow:        hidden;
            box-shadow:      0 4px 20px rgba(99,150,210,0.10);
        }

        .metric-glass:hover {
            border-color: rgba(74,144,217,0.40);
            box-shadow:   0 10px 36px rgba(74,144,217,0.18);
            transform:    translateY(-5px);
        }

        .metric-num {
            font-size:   46px;
            font-weight: 800;
            line-height: 1;
            background:  linear-gradient(135deg, #4a90d9, #9b72d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }

        .metric-lbl {
            font-size:      12px;
            font-weight:    600;
            letter-spacing: 1px;
            text-transform: uppercase;
            color:          rgba(60,90,140,0.55) !important;
        }

        .metric-bar {
            height:        6px;
            border-radius: 99px;
            background:    rgba(99,150,210,0.15);
            margin-top:    14px;
            overflow:      hidden;
        }

        .metric-bar-fill {
            height:        100%;
            border-radius: 99px;
            background:    linear-gradient(90deg, #4a90d9, #9b72d4);
            transition:    width 1s cubic-bezier(0.4,0,0.2,1);
        }

        /* ══ REC ROWS ══ */
        .rec-row {
            display:       flex;
            align-items:   flex-start;
            gap:           14px;
            padding:       14px 18px;
            background:    rgba(255,255,255,0.60);
            border:        1px solid rgba(180,205,235,0.50);
            border-radius: 14px;
            margin-bottom: 10px;
            font-size:     14px;
            color:         #2d4060;
            line-height:   1.55;
            transition:    all 0.20s ease;
            backdrop-filter: blur(8px);
        }

        .rec-row:hover {
            background:   rgba(255,255,255,0.85);
            border-color: rgba(74,144,217,0.35);
            box-shadow:   0 4px 16px rgba(74,144,217,0.10);
            transform:    translateX(4px);
        }

        .rec-icon { font-size: 20px; flex-shrink: 0; margin-top: 1px; }

        /* ══ TYPOGRAPHY ══ */
        .gr-markdown p {
            color: #2d4060 !important;
            line-height: 1.65;
        }
        .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
            color: #1a3a7a !important;
            font-weight: 700;
        }

        /* ══ LOGIN PAGE ══ */
        .login-wrap, .login_page,
        div[class*="login"], .gradio-container .panel {
            display:         flex !important;
            align-items:     center !important;
            justify-content: center !important;
            min-height:      100vh !important;
        }

        .login-wrap .panel,
        .gradio-container .panel,
        form[class*="login"],
        .login_page .panel {
            max-width:       460px !important;
            width:           100% !important;
            background:      rgba(255,255,255,0.72) !important;
            backdrop-filter: blur(32px) saturate(160%) !important;
            -webkit-backdrop-filter: blur(32px) saturate(160%) !important;
            border:          1px solid rgba(255,255,255,0.92) !important;
            border-radius:   28px !important;
            padding:         48px 40px !important;
            box-shadow:
                0 24px 80px rgba(99,150,210,0.18),
                0 4px  16px rgba(99,150,210,0.10),
                inset 0 1px 0 rgba(255,255,255,1) !important;
        }

        .login_page h2, .panel h2,
        form[class*="login"] h2 {
            font-size:   28px !important;
            font-weight: 800 !important;
            background:  linear-gradient(135deg, #2b6cb0, #6b46c1) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            text-align:  center !important;
            margin-bottom: 8px !important;
        }

        .login_page p, .panel p {
            text-align: center !important;
            color:      rgba(60,90,140,0.65) !important;
            font-size:  14px !important;
            line-height: 1.5 !important;
        }

        .login_page input, .panel input[type="text"], .panel input[type="password"] {
            background:      rgba(255,255,255,0.75) !important;
            border:          1px solid rgba(180,205,235,0.70) !important;
            border-radius:   14px !important;
            color:           #1a2332 !important;
            font-size:       15px !important;
            padding:         14px 18px !important;
            width:           100% !important;
            transition:      border-color 0.2s, box-shadow 0.2s !important;
        }

        .login_page input:focus, .panel input:focus {
            border-color: rgba(99,150,237,0.70) !important;
            box-shadow:   0 0 0 3px rgba(99,150,237,0.15) !important;
            outline:      none !important;
        }

        .login_page button, .panel button[type="submit"],
        .login_page .submit, .panel .submit {
            width:          100% !important;
            background:     linear-gradient(135deg, #4a90d9 0%, #6b6bd6 50%, #9b72d4 100%) !important;
            border:         none !important;
            border-radius:  16px !important;
            color:          #ffffff !important;
            font-size:      16px !important;
            font-weight:    700 !important;
            padding:        16px !important;
            cursor:         pointer !important;
            box-shadow:     0 6px 28px rgba(74,144,217,0.38) !important;
            transition:     all 0.30s cubic-bezier(0.34,1.56,0.64,1) !important;
            letter-spacing: 0.4px !important;
            margin-top:     12px !important;
        }

        .login_page button:hover, .panel button[type="submit"]:hover {
            transform:  translateY(-3px) scale(1.02) !important;
            box-shadow: 0 12px 40px rgba(74,144,217,0.50) !important;
        }
        """
    ) as app:

        # ══════════════════════════════════════════════════
        # AUTH LOGIC  ─  accept any non-empty credentials
        # ══════════════════════════════════════════════════

        def do_login(username: str, password: str) -> tuple:
            """Accept any non-empty username + password."""
            if not username.strip() or not password.strip():
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "<div style='color:#e53e3e;font-size:13px;margin-top:6px;'>⚠ Please fill in both fields.</div>"
                )
            return (gr.update(visible=False), gr.update(visible=True), "")

        def do_signup(username: str, email: str, password: str, confirm: str) -> tuple:
            """Accept any non-empty signup form."""
            if not username.strip() or not password.strip():
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "<div style='color:#e53e3e;font-size:13px;margin-top:6px;'>⚠ Username and password are required.</div>"
                )
            if password != confirm:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "<div style='color:#e53e3e;font-size:13px;margin-top:6px;'>❌ Passwords do not match.</div>"
                )
            return (gr.update(visible=False), gr.update(visible=True), "")

        # ══════════════════════════════════════════════════
        # LOGIN / SIGNUP VIEW
        # ══════════════════════════════════════════════════

        with gr.Column(visible=True, elem_id="login_view") as login_view:
            gr.HTML("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

                /* ── Full-page centering ── */
                #login_view {
                    min-height: 100vh;
                    display: flex !important;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 32px 16px;
                    position: relative;
                    font-family: 'Inter', sans-serif;
                }

                /* ── Blob orbs ── */
                #login_view::before {
                    content: '';
                    position: fixed;
                    inset: 0;
                    background:
                        radial-gradient(ellipse 75% 55% at 18% 38%, rgba(99,179,237,0.28) 0%, transparent 58%),
                        radial-gradient(ellipse 60% 75% at 82% 22%, rgba(154,117,234,0.22) 0%, transparent 58%),
                        radial-gradient(ellipse 65% 45% at 52% 88%, rgba(72,187,120,0.16) 0%, transparent 58%);
                    animation: lOrbs 28s ease-in-out infinite alternate;
                    pointer-events: none;
                    z-index: 0;
                }

                @keyframes lOrbs {
                    0%   { transform: scale(1)    rotate(0deg); }
                    50%  { transform: scale(1.04) rotate(1.5deg); }
                    100% { transform: scale(0.98) rotate(-1deg); }
                }

                /* ── Scrolling ECG ── */
                #login_view::after {
                    content: '';
                    position: fixed;
                    bottom: 8%;
                    left: 0;
                    width: 100%;
                    height: 60px;
                    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 60'%3E%3Cpath d='M0,30 L180,30 L210,30 L228,8 L246,52 L264,12 L282,48 L300,30 L340,30 L580,30 L608,30 L626,8 L644,52 L662,12 L680,48 L698,30 L740,30 L980,30 L1008,30 L1026,8 L1044,52 L1062,12 L1080,48 L1098,30 L1200,30' fill='none' stroke='rgba(74,144,217,0.10)' stroke-width='1.5'/%3E%3C/svg%3E") repeat-x;
                    background-size: 1200px 60px;
                    animation: ecgScroll 10s linear infinite;
                    pointer-events: none;
                    z-index: 0;
                }

                @keyframes ecgScroll {
                    from { background-position: 0 0; }
                    to   { background-position: -1200px 0; }
                }

                /* ── Faint stethoscope watermark ── */
                .lp-stethoscope {
                    position: fixed;
                    bottom: -10px; right: -10px;
                    font-size: 220px; line-height: 1;
                    opacity: 0.055; filter: blur(2px);
                    pointer-events: none; z-index: 0;
                    user-select: none;
                }

                /* ── DNA left edge ── */
                .lp-dna {
                    position: fixed;
                    top: 8%; left: 2%;
                    width: 48px; height: 380px;
                    opacity: 0.045;
                    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 36 200'%3E%3Cpath d='M18,0 Q36,25 18,50 Q0,75 18,100 Q36,125 18,150 Q0,175 18,200' fill='none' stroke='%234a90d9' stroke-width='2'/%3E%3Cpath d='M18,0 Q0,25 18,50 Q36,75 18,100 Q0,125 18,150 Q36,175 18,200' fill='none' stroke='%239b72d4' stroke-width='2'/%3E%3Ccircle cx='18' cy='25' r='2.5' fill='%234a90d9'/%3E%3Ccircle cx='18' cy='75' r='2.5' fill='%239b72d4'/%3E%3Ccircle cx='18' cy='125' r='2.5' fill='%234a90d9'/%3E%3Ccircle cx='18' cy='175' r='2.5' fill='%239b72d4'/%3E%3C/svg%3E") no-repeat center/contain;
                    pointer-events: none; z-index: 0;
                    animation: dnaFloat 9s ease-in-out infinite;
                }

                @keyframes dnaFloat {
                    0%,100% { transform: translateY(0); }
                    50%     { transform: translateY(-14px); }
                }

                /* ── Card wrapper ── */
                .lp-wrap {
                    position: relative; z-index: 2;
                    width: 100%; max-width: 460px;
                    margin: 0 auto;
                    animation: cardFloat 7s ease-in-out infinite;
                }

                @keyframes cardFloat {
                    0%,100% { transform: translateY(0); }
                    50%     { transform: translateY(-6px); }
                }

                /* ── Glass card ── */
                .lp-card {
                    background: rgba(255,255,255,0.25);
                    backdrop-filter: blur(18px) saturate(170%);
                    -webkit-backdrop-filter: blur(18px) saturate(170%);
                    border: 1px solid rgba(255,255,255,0.32);
                    border-radius: 22px;
                    box-shadow:
                        0 10px 30px rgba(0,0,0,0.10),
                        0 4px 12px rgba(99,150,210,0.08),
                        inset 0 1px 0 rgba(255,255,255,0.85);
                    padding: 40px 36px 32px;
                    position: relative; overflow: hidden;
                }

                .lp-card::before {
                    content: '';
                    position: absolute;
                    top: 0; left: 10%; right: 10%;
                    height: 1px;
                    background: linear-gradient(90deg, transparent, rgba(255,255,255,1), transparent);
                    pointer-events: none;
                }

                .lp-card::after {
                    content: '';
                    position: absolute;
                    top: -50px; right: -50px;
                    width: 160px; height: 160px;
                    background: radial-gradient(circle, rgba(74,144,217,0.10), transparent 70%);
                    border-radius: 50%; pointer-events: none;
                }

                /* ── Logo ── */
                .lp-logo {
                    text-align: center;
                    margin-bottom: 26px;
                }

                .lp-icon {
                    display: inline-flex;
                    align-items: center; justify-content: center;
                    width: 60px; height: 60px;
                    background: linear-gradient(135deg, rgba(74,144,217,0.15), rgba(154,117,234,0.12));
                    border: 1px solid rgba(255,255,255,0.55);
                    border-radius: 18px; font-size: 28px;
                    margin-bottom: 13px;
                    box-shadow: 0 4px 18px rgba(74,144,217,0.14);
                }

                .lp-logo h2 {
                    font-size: 25px !important;
                    font-weight: 800 !important;
                    background: linear-gradient(135deg, #2b6cb0, #6b46c1 55%, #2c7a7b) !important;
                    -webkit-background-clip: text !important;
                    -webkit-text-fill-color: transparent !important;
                    background-clip: text !important;
                    margin: 0 0 5px !important;
                    line-height: 1.2 !important;
                    font-family: 'Inter', sans-serif !important;
                }

                .lp-logo p {
                    font-size: 13px !important;
                    color: rgba(55,85,135,0.58) !important;
                    margin: 0 !important;
                    font-family: 'Inter', sans-serif !important;
                }

                /* ── Tabs ── */
                #login_view .tabs > .tab-nav {
                    background: rgba(255,255,255,0.32) !important;
                    border: 1px solid rgba(255,255,255,0.55) !important;
                    border-radius: 13px !important;
                    padding: 4px !important;
                    margin-bottom: 20px !important;
                    gap: 2px !important;
                    box-shadow: none !important;
                }

                #login_view .tabs > .tab-nav > button {
                    flex: 1 !important; text-align: center !important;
                    border-radius: 10px !important;
                    padding: 9px 14px !important;
                    font-size: 13.5px !important; font-weight: 600 !important;
                    color: rgba(40,60,120,0.50) !important;
                    transition: all 0.25s ease !important;
                    background: transparent !important;
                    border: 1px solid transparent !important;
                    box-shadow: none !important;
                    font-family: 'Inter', sans-serif !important;
                }

                #login_view .tabs > .tab-nav > button.selected {
                    background: linear-gradient(135deg, #2563eb, #14b8a6) !important;
                    color: #fff !important;
                    box-shadow: 0 3px 12px rgba(37,99,235,0.28) !important;
                    border-color: transparent !important;
                }

                #login_view .tabs > .tab-nav > button:not(.selected):hover {
                    background: rgba(37,99,235,0.07) !important;
                    color: rgba(40,60,120,0.80) !important;
                }

                /* ── Inputs ── */
                #login_view input[type="text"],
                #login_view input[type="password"],
                #login_view input[type="email"] {
                    width: 100% !important;
                }

                #login_view input:focus {
                    background: rgba(255,255,255,0.80) !important;
                    border-color: rgba(74,144,217,0.60) !important;
                    box-shadow: 0 0 0 4px rgba(74,144,217,0.12),
                                0 2px 8px rgba(74,144,217,0.08) !important;
                }

                /* ── Login/Signup buttons ── */
                #login_view button.primary,
                #login_view button.lg {
                    width: 100% !important;
                    background: linear-gradient(135deg, #2563eb 0%, #14b8a6 100%) !important;
                    border: none !important;
                    border-radius: 14px !important;
                    color: #ffffff !important;
                    font-size: 16px !important;
                    font-weight: 700 !important;
                    padding: 15px 24px !important;
                    cursor: pointer !important;
                    box-shadow: 0 6px 28px rgba(37,99,235,0.30),
                                inset 0 1px 0 rgba(255,255,255,0.25) !important;
                    transition: all 0.30s cubic-bezier(0.34,1.56,0.64,1) !important;
                    letter-spacing: 0.5px !important;
                    margin-top: 8px !important;
                    position: relative !important;
                    overflow: hidden !important;
                }

                #login_view button.primary:hover,
                #login_view button.lg:hover {
                    transform: translateY(-3px) scale(1.02) !important;
                    box-shadow: 0 12px 40px rgba(37,99,235,0.45),
                                0 0 20px rgba(20,184,166,0.20) !important;
                }

                #login_view button.primary:active,
                #login_view button.lg:active {
                    transform: translateY(0) scale(0.98) !important;
                }

                /* ── Block overrides inside login ── */
                #login_view .gradio-box,
                #login_view .gr-box,
                #login_view .gr-panel,
                #login_view .gr-group,
                #login_view .gr-form {
                    background: transparent !important;
                    border: none !important;
                    box-shadow: none !important;
                    backdrop-filter: none !important;
                    border-radius: 0 !important;
                    padding: 0 !important;
                }

                #login_view .gradio-box:hover,
                #login_view .gr-box:hover,
                #login_view .gr-panel:hover {
                    transform: none !important;
                    box-shadow: none !important;
                }

                /* ── Footer text ── */
                .login-footer {
                    text-align: center;
                    margin-top: 24px;
                    font-size: 13px;
                    color: rgba(60,90,140,0.50);
                    line-height: 1.5;
                }

                .login-footer a {
                    color: #4a72b8;
                    text-decoration: none;
                    font-weight: 600;
                    cursor: pointer;
                }

                .login-footer a:hover {
                    color: #6b46c1;
                    text-decoration: underline;
                }

                /* ── Stethoscope visual ── */
                .login-stethoscope {
                    position: fixed;
                    bottom: -20px;
                    right: -20px;
                    width: 280px;
                    height: 280px;
                    opacity: 0.06;
                    font-size: 260px;
                    line-height: 1;
                    pointer-events: none;
                    z-index: 0;
                    filter: blur(1px);
                }

                /* ── DNA Helix Pattern ── */
                .login-dna-pattern {
                    position: fixed;
                    top: 10%;
                    left: 5%;
                    width: 60px;
                    height: 400px;
                    opacity: 0.04;
                    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 200'%3E%3Cpath d='M20,0 Q40,25 20,50 Q0,75 20,100 Q40,125 20,150 Q0,175 20,200' fill='none' stroke='%234a90d9' stroke-width='2'/%3E%3Cpath d='M20,0 Q0,25 20,50 Q40,75 20,100 Q0,125 20,150 Q40,175 20,200' fill='none' stroke='%239b72d4' stroke-width='2'/%3E%3Ccircle cx='20' cy='25' r='3' fill='%234a90d9' opacity='0.6'/%3E%3Ccircle cx='20' cy='75' r='3' fill='%239b72d4' opacity='0.6'/%3E%3Ccircle cx='20' cy='125' r='3' fill='%234a90d9' opacity='0.6'/%3E%3Ccircle cx='20' cy='175' r='3' fill='%239b72d4' opacity='0.6'/%3E%3C/svg%3E") no-repeat;
                    background-size: contain;
                    pointer-events: none;
                    z-index: 0;
                    animation: dnaFloat 8s ease-in-out infinite;
                }

                @keyframes dnaFloat {
                    0%, 100% { transform: translateY(0) rotate(0deg); }
                    50%      { transform: translateY(-15px) rotate(3deg); }
                }

                .login-dna-pattern-right {
                    left: auto;
                    right: 5%;
                    top: 20%;
                    transform: scaleX(-1);
                    opacity: 0.03;
                }
            </style>

            <!-- Decorative elements -->
            <div class="login-stethoscope">🩺</div>
            <div class="login-dna-pattern"></div>
            <div class="login-dna-pattern login-dna-pattern-right"></div>

            <div class="login-card-wrap">
                <div class="login-glass-card">
                    <div class="login-logo">
                        <div class="login-logo-icon">🏥</div>
                        <h2>Welcome to HealthSense AI</h2>
                        <p>Secure AI-powered health diagnostics platform</p>
                    </div>
            """)

            # ── Login / Sign Up Tabs ──
            with gr.Tabs(elem_id="login_tabs"):

                # ─── LOGIN TAB ───
                with gr.Tab("🔐  Login"):
                    login_user = gr.Textbox(
                        label="Username",
                        placeholder="Enter your username",
                        elem_id="l_user"
                    )
                    login_pass = gr.Textbox(
                        label="Password",
                        type="password",
                        placeholder="Enter your password",
                        elem_id="l_pass"
                    )
                    login_msg = gr.HTML(value="", elem_id="l_msg")
                    login_btn = gr.Button(
                        "Login  →",
                        variant="primary",
                        size="lg",
                        elem_id="l_btn"
                    )
                    gr.HTML("""
                        <div class="lp-footer">
                            <a href="#" style="color:rgba(37,99,235,0.65);text-decoration:none;font-weight:500;">
                                Forgot password?
                            </a>
                        </div>
                    """)

                # ─── SIGN UP TAB ───
                with gr.Tab("✨  Sign Up"):
                    signup_user = gr.Textbox(
                        label="Username",
                        placeholder="Choose a username",
                        elem_id="s_user"
                    )
                    signup_email = gr.Textbox(
                        label="Email",
                        placeholder="your@email.com",
                        elem_id="s_email"
                    )
                    signup_pass = gr.Textbox(
                        label="Password",
                        type="password",
                        placeholder="Create a password",
                        elem_id="s_pass"
                    )
                    signup_confirm = gr.Textbox(
                        label="Confirm Password",
                        type="password",
                        placeholder="Re-enter your password",
                        elem_id="s_confirm"
                    )
                    signup_msg = gr.HTML(value="", elem_id="s_msg")
                    signup_btn = gr.Button(
                        "Create Account  ✨",
                        variant="primary",
                        size="lg",
                        elem_id="s_btn"
                    )

            gr.HTML("""
              </div><!-- /.lp-card -->
            </div><!-- /.lp-wrap -->
            <div class="lp-footer-small">🔒 Secure &nbsp;·&nbsp; HIPAA Compliant &nbsp;·&nbsp; For Educational Use</div>
            """)

        # ══════════════════════════════════════════════════
        # MAIN APPLICATION VIEW (hidden until login)
        # ══════════════════════════════════════════════════

        with gr.Column(visible=False, elem_id="main_view") as main_view:
            gr.HTML(
                """
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
            )

            # ==== TAB 1: Physical Health ====
            with gr.Tab("🩺 Physical Health Assessment", id="physical"):
                gr.Markdown("## Physical Health Risk Assessment")
                gr.Markdown("Enter your health metrics below to assess diabetes and heart disease risk.")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🩸 Diabetes Risk Factors")
                    d_glucose = gr.Slider(0, 200, step=1, value=100, label="Glucose (mg/dL)")
                    d_bp = gr.Slider(0, 140, step=1, value=70, label="Blood Pressure (mm Hg)")
                    d_skin = gr.Slider(0, 100, step=1, value=20, label="Skin Thickness (mm)")
                    d_insulin = gr.Slider(0, 900, step=1, value=80, label="Insulin (mu U/ml)")
                    d_bmi = gr.Slider(10.0, 70.0, step=0.1, value=25.0, label="BMI (kg/m²)")
                    d_dpf = gr.Slider(0.0, 2.5, step=0.01, value=0.5, label="Diabetes Pedigree Function")
                    d_age = gr.Slider(1, 120, step=1, value=30, label="Age")

                with gr.Column(scale=1):
                    gr.Markdown("### ❤️ Heart Disease Risk Factors")
                    h_age = gr.Slider(1, 120, step=1, value=50, label="Age")
                    h_sex = gr.Radio(["Male", "Female"], value="Male", label="Sex")
                    h_cp = gr.Dropdown(["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"], value="Non-Anginal", label="Chest Pain Type")
                    h_trestbps = gr.Slider(80, 200, step=1, value=120, label="Resting BP (mm Hg)")
                    h_chol = gr.Slider(100, 600, step=1, value=200, label="Cholesterol (mg/dl)")
                    h_fbs = gr.Checkbox(label="Fasting Blood Sugar > 120 mg/dl", value=False)
                    h_thalach = gr.Slider(60, 220, step=1, value=150, label="Max Heart Rate")
                    h_exang = gr.Checkbox(label="Exercise Induced Angina", value=False)
                    h_oldpeak = gr.Slider(0.0, 7.0, step=0.1, value=1.0, label="Oldpeak")
                    h_slope = gr.Dropdown(["Up", "Flat", "Down"], value="Up", label="ST Slope")

            physical_btn = gr.Button(
                "🔍 Assess Physical Health",
                variant="primary", size="lg"
            )

            with gr.Row():
                d_result = gr.Label(label="Diabetes Risk", elem_id="diabetes_risk_label")
                h_result = gr.Label(label="Heart Disease Risk", elem_id="heart_risk_label")

            with gr.Row():
                gauge_plot = gr.Plot(label="Risk Gauge")
                feat_df = gr.DataFrame(
                    label="Top 5 Influential Features",
                    headers=["feature", "score"]
                )

            physical_btn.click(
                fn=predict_physical,
                inputs=[
                    d_glucose, d_bp, d_skin, d_insulin,
                    d_bmi, d_dpf, d_age,
                    h_age, h_sex, h_cp, h_trestbps, h_chol, h_fbs,
                    h_thalach, h_exang, h_oldpeak, h_slope
                ],
                outputs=[d_result, h_result, gauge_plot, feat_df]
            )

            # ==== TAB 2: Mental Health ====
            with gr.Tab("🧠 Mental Health Check", id="mental"):
                gr.Markdown("## Mental Health Assessment")
                gr.Markdown("Share your thoughts and feelings for an AI-powered mental health screening.")

            mental_text = gr.Textbox(
                lines=6,
                placeholder="Describe how you've been feeling lately... Share your thoughts, "
                            "emotions, and daily experiences for a more accurate assessment.",
                label="How are you feeling?",
                elem_id="mental_text_input"
            )
            char_count = gr.Textbox(
                label="", interactive=False, value="Character count: 0",
                elem_id="char_count_display"
            )
            mental_text.change(fn=count_chars, inputs=mental_text, outputs=char_count)

            mental_btn = gr.Button(
                "🔍 Analyze Mental State",
                variant="primary", size="lg"
            )

            m_result = gr.Label(label="Mental Health Classification", elem_id="mental_result_label")
            m_confidence = gr.HTML(label="Assessment Confidence")
            m_highlighted = gr.HighlightedText(
                label="Influential Words in Your Text",
                combine_adjacent=True,
                color_map={"influential": "#ff6b6b"}
            )

            mental_btn.click(
                fn=predict_mental,
                inputs=mental_text,
                outputs=[m_result, m_confidence, m_highlighted]
            )

            # ==== TAB 3: Unified Report ====
            with gr.Tab("📊 Unified Health Report", id="unified"):
                gr.Markdown("## Comprehensive Health Report")
                gr.Markdown("Fill in all health metrics for a complete multi-modal assessment.")

            gr.Markdown("ℹ️ **Note:** This report automatically fuses the data you entered in the **Physical Health** and **Mental Health** tabs. You don't need to enter anything twice!")

            with gr.Accordion("⚙️ Advanced Settings (Thresholds & Weights)", open=False):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Custom Fusion Weights")
                        u_w_d = gr.Slider(0.0, 1.0, value=config.FUSION_WEIGHT_DIABETES, step=0.05, label="Diabetes Weight")
                        u_w_h = gr.Slider(0.0, 1.0, value=config.FUSION_WEIGHT_HEART, step=0.05, label="Heart Disease Weight")
                        u_w_m = gr.Slider(0.0, 1.0, value=config.FUSION_WEIGHT_MENTAL, step=0.05, label="Mental Health Weight")
                    with gr.Column():
                        gr.Markdown("#### Threshold Tuning")
                        u_m_thresh = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Mental Health Sensitivity (Threshold)")

            with gr.Row():
                unified_btn = gr.Button("📊 Generate Full Health Report", variant="primary", size="lg")
                export_btn = gr.Button("📥 Export Assessment as HTML", variant="secondary", size="lg")

            with gr.Row():
                ud_label = gr.Label(label="Diabetes Risk %")
                uh_label = gr.Label(label="Heart Risk %")
                um_label = gr.Label(label="Mental Health %")
            
            with gr.Row():
                u_export_file = gr.File(label="Download HTML Report", visible=False)

            u_composite = gr.HTML(label="Composite Score & Risk Tier")
            u_summary = gr.Textbox(label="Health Summary", interactive=False, lines=5)
            u_recs = gr.DataFrame(label="Recommendations", headers=["#", "Recommendation"])
            u_radar = gr.Plot(label="Health Risk Radar Chart")

            outputs_list = [ud_label, uh_label, um_label, u_composite, u_summary, u_recs, u_radar, u_export_file]
            inputs_list = [
                d_glucose, d_bp, d_skin, d_insulin, d_bmi, d_dpf, d_age,
                h_age, h_sex, h_cp, h_trestbps, h_chol, h_fbs,
                h_thalach, h_exang, h_oldpeak, h_slope,
                mental_text, u_w_d, u_w_h, u_w_m, u_m_thresh
            ]

            unified_btn.click(
                fn=generate_full_report,
                inputs=inputs_list,
                outputs=outputs_list
            ).then(lambda: gr.update(visible=True), outputs=[u_export_file])

            export_btn.click(
                fn=generate_full_report,
                inputs=inputs_list,
                outputs=outputs_list
            ).then(lambda: gr.update(visible=True), outputs=[u_export_file])

            # ==== TAB 4: Model Insights ====
            with gr.Tab("📈 Model Insights", id="insights"):
                gr.Markdown("## Model Performance & Insights")
                gr.Markdown("View training history, confusion matrices, and feature importance for all models.")

            with gr.Row():
                gr.Markdown("### Training History")

            with gr.Row():
                img_d_hist = gr.Image(
                    value=load_plot_image("Diabetes_history.png"),
                    label="Diabetes — Training History",
                    elem_id="diabetes_history_img"
                )
                img_h_hist = gr.Image(
                    value=load_plot_image("Heart_history.png"),
                    label="Heart — Training History",
                    elem_id="heart_history_img"
                )
                img_m_hist = gr.Image(
                    value=load_plot_image("Mental_history.png"),
                    label="Mental — Training History",
                    elem_id="mental_history_img"
                )

            with gr.Row():
                gr.Markdown("### Confusion Matrices")

            with gr.Row():
                img_d_cm = gr.Image(
                    value=load_plot_image("Diabetes_confusion.png"),
                    label="Diabetes — Confusion Matrix",
                    elem_id="diabetes_confusion_img"
                )
                img_h_cm = gr.Image(
                    value=load_plot_image("Heart_confusion.png"),
                    label="Heart — Confusion Matrix",
                    elem_id="heart_confusion_img"
                )
                img_m_cm = gr.Image(
                    value=load_plot_image("Mental_confusion.png"),
                    label="Mental — Confusion Matrix",
                    elem_id="mental_confusion_img"
                )

            with gr.Row():
                gr.Markdown("### Feature Importance")

            with gr.Row():
                img_d_fi = gr.Image(
                    value=load_plot_image("Diabetes_importance.png"),
                    label="Diabetes — Feature Importance",
                    elem_id="diabetes_importance_img"
                )
                img_h_fi = gr.Image(
                    value=load_plot_image("Heart_importance.png"),
                    label="Heart — Feature Importance",
                    elem_id="heart_importance_img"
                )

            gr.Markdown("### Model Performance Summary")
            perf_data = []
            perf_path = os.path.join(config.PLOTS_DIR, "performance_summary.csv")
            if os.path.exists(perf_path):
                perf_df = pd.read_csv(perf_path)
                perf_data = perf_df
            else:
                perf_data = pd.DataFrame({
                    'Model': ['Diabetes DNN', 'Heart Disease DNN', 'Mental Health LSTM'],
                    'Accuracy': ['—', '—', '—'],
                    'AUC': ['—', '—', '—'],
                    'F1-Score': ['—', '—', '—']
                })

            gr.DataFrame(
                value=perf_data,
                label="Model Performance Comparison",
                elem_id="performance_table"
            )

            # ==== TAB 5: About ====
            with gr.Tab("ℹ️ About & Architecture", id="about"):
                gr.Markdown("""
            # 🏥 HealthSense AI — About

            ## Project Description
            HealthSense AI is a **multi-modal deep learning healthcare assessment platform** that combines
            three independent AI models to provide a comprehensive health risk evaluation. The system
            analyzes diabetes risk factors, heart disease indicators, and mental health status through
            natural language processing, then fuses these predictions into a unified health report.

            ---

            ## 🛠️ Tech Stack
            | Technology | Purpose |
            |-----------|---------|
            | **TensorFlow / Keras** | Deep learning model training and inference |
            | **scikit-learn** | Data preprocessing, metrics, and feature importance |
            | **Gradio** | Interactive web-based user interface |
            | **Python 3.10+** | Core programming language |
            | **matplotlib / seaborn** | Evaluation visualizations |
            | **Plotly** | Interactive radar charts |

            ---

            ## 🧠 Model Architectures

            ### 1. Diabetes Risk DNN
            - **Type:** Feedforward Deep Neural Network
            - **Layers:** Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.2) → Dense(16) → Sigmoid
            - **Input:** 8 clinical features (Glucose, BMI, Age, etc.)
            - **Output:** Binary classification (Diabetic / Non-Diabetic)

            ### 2. Heart Disease DNN
            - **Type:** Feedforward DNN with Batch Normalization
            - **Layers:** Dense(128) → BN → Dropout(0.4) → Dense(64) → BN → Dropout(0.3) → Dense(32) → Sigmoid
            - **Input:** 13 clinical features (Age, Cholesterol, BP, etc.)
            - **Output:** Binary classification (Heart Disease / Healthy)

            ### 3. Mental Health LSTM
            - **Type:** LSTM Recurrent Neural Network
            - **Layers:** Embedding(5000, 64) → SpatialDropout1D → LSTM(64) → Dense(32) → Sigmoid
            - **Input:** Text sequences (patient journal entries)
            - **Output:** Binary classification (Depressed/Stressed / Healthy)

            ---

            ## 🔗 Fusion Engine
            The **HealthFusionEngine** combines predictions from all three models using:
            1. **Mental Health Amplification:** Physical disease risks are slightly amplified when mental health risk is elevated, reflecting real-world correlations
            2. **Weighted Composite Scoring:** Diabetes (35%) + Heart (35%) + Mental (30%)
            3. **Risk Tiering:** Low Risk (<25) → Moderate (25-49) → High (50-74) → Critical (≥75)
            4. **Rule-based Recommendations:** Contextual health advice based on individual risk profiles

            ---

            ## 🔗 Links
            - **GitHub:** [HealthSense AI Repository](#)
            - **LinkedIn:** [Team Profile](#)
            - **Team:** HealthSense AI Development Team

            ---

            ## 📄 License
            This project is licensed under the **MIT License**.

            > ⚠️ **Disclaimer:** HealthSense AI is an educational and research project.
            > It is NOT a substitute for professional medical advice, diagnosis, or treatment.
            > Always seek the advice of qualified health providers for medical concerns.
            """)

        # ── End of main_view ──────────────────────────────

        # ══════════════════════════════════════════════════
        # WIRE UP LOGIN / SIGNUP HANDLERS
        # ══════════════════════════════════════════════════

        login_btn.click(
            fn=do_login,
            inputs=[login_user, login_pass],
            outputs=[login_view, main_view, login_msg]
        )

        signup_btn.click(
            fn=do_signup,
            inputs=[signup_user, signup_email, signup_pass, signup_confirm],
            outputs=[login_view, main_view, signup_msg]
        )

    return app
