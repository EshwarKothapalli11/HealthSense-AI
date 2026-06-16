"""
fusion_engine.py — Multi-modal health risk fusion engine for HealthSense AI.

Combines diabetes, heart disease, and mental health predictions into a unified
health assessment with risk tiers, summaries, and recommendations.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config


class HealthFusionEngine:
    """
    Fuses predictions from three health models into a comprehensive assessment.

    Applies mental health amplification to disease risk scores and generates
    unified risk tiers, health summaries, and actionable recommendations.
    """

    def __init__(
        self,
        w_diabetes: float = 0.35,
        w_heart: float = 0.35,
        w_mental: float = 0.30
    ) -> None:
        """
        Initialize the fusion engine with model weights.

        Args:
            w_diabetes: Weight for diabetes risk in composite score.
            w_heart: Weight for heart disease risk in composite score.
            w_mental: Weight for mental health risk in composite score.
        """
        self.weights = {
            'diabetes': w_diabetes,
            'heart': w_heart,
            'mental': w_mental
        }

    def amplify_risk(
        self, disease_prob: float, mental_prob: float, alpha: float = 0.12
    ) -> float:
        """
        Amplify disease risk based on mental health status.

        If mental health risk is elevated, slightly increases the disease risk
        to model the well-documented correlation between mental and physical health.

        Formula: min(1.0, disease_prob + alpha * mental_prob * disease_prob)

        Args:
            disease_prob: Raw disease probability [0, 1].
            mental_prob: Mental health risk probability [0, 1].
            alpha: Amplification factor controlling mental health influence.

        Returns:
            Adjusted disease probability, capped at 1.0.
        """
        return min(1.0, disease_prob + alpha * mental_prob * disease_prob)

    def fuse(
        self, diabetes_prob: float, heart_prob: float, mental_prob: float,
        w_diabetes: float = None, w_heart: float = None, w_mental: float = None,
        m_threshold: float = 0.5
    ) -> dict:
        """
        Fuse all three model predictions into a unified health assessment.

        Args:
            diabetes_prob: Diabetes risk probability [0, 1].
            heart_prob: Heart disease risk probability [0, 1].
            mental_prob: Mental health risk probability [0, 1].

        Returns:
            Dictionary containing:
                - diabetes_risk_pct: Adjusted diabetes risk percentage
                - heart_risk_pct: Adjusted heart risk percentage
                - mental_status: Depression/stress status string
                - mental_confidence_pct: Mental health confidence percentage
                - composite_score: Weighted composite health score (0-100)
                - risk_tier: Overall risk category
                - health_summary: Natural language health summary
                - recommendations: List of actionable health recommendations
        """
        # 1. Apply mental health amplification to disease risks
        diabetes_adj = self.amplify_risk(
            diabetes_prob, mental_prob, alpha=config.MENTAL_AMPLIFICATION_ALPHA
        )
        heart_adj = self.amplify_risk(
            heart_prob, mental_prob, alpha=config.MENTAL_AMPLIFICATION_ALPHA
        )

        # 2. Determine weights
        w_d = w_diabetes if w_diabetes is not None else self.weights['diabetes']
        w_h = w_heart if w_heart is not None else self.weights['heart']
        w_m = w_mental if w_mental is not None else self.weights['mental']
        
        total_w = w_d + w_h + w_m
        if total_w > 0:
            w_d, w_h, w_m = w_d/total_w, w_h/total_w, w_m/total_w

        # 3. Compute composite score (weighted average, scaled to 0-100)
        composite = (
            w_d * diabetes_adj +
            w_h * heart_adj +
            w_m * mental_prob
        )

        # 3. Determine risk tier
        if composite < 0.25:
            risk_tier = 'Low Risk'
        elif composite < 0.50:
            risk_tier = 'Moderate Risk'
        elif composite < 0.75:
            risk_tier = 'High Risk'
        else:
            risk_tier = 'Critical'

        # 4. Mental health status
        mental_status = 'Depressed/Stressed' if mental_prob >= m_threshold else 'Healthy'

        # 5. Generate summary and recommendations
        health_summary = self._generate_summary(
            diabetes_adj, heart_adj, mental_prob, risk_tier
        )
        recommendations = self._generate_recommendations(
            diabetes_adj, heart_adj, mental_prob
        )

        return {
            'diabetes_risk_pct': round(diabetes_adj * 100, 1),
            'heart_risk_pct': round(heart_adj * 100, 1),
            'mental_status': mental_status,
            'mental_confidence_pct': round(mental_prob * 100, 1),
            'composite_score': round(composite * 100, 1),
            'risk_tier': risk_tier,
            'health_summary': health_summary,
            'recommendations': recommendations
        }

    def _generate_summary(
        self, d: float, h: float, m: float, tier: str
    ) -> str:
        """
        Generate a natural language health summary.

        Args:
            d: Adjusted diabetes risk probability.
            h: Adjusted heart risk probability.
            m: Mental health risk probability.
            tier: Computed risk tier string.

        Returns:
            Comprehensive health summary paragraph.
        """
        d_pct = round(d * 100, 1)
        h_pct = round(h * 100, 1)
        m_pct = round(m * 100, 1)

        summary_parts = []

        # Opening statement
        summary_parts.append(
            f"Based on our comprehensive multi-modal health analysis, "
            f"your overall health risk tier is: **{tier}**."
        )

        # Diabetes assessment
        if d >= 0.7:
            summary_parts.append(
                f"Your diabetes risk is significantly elevated at {d_pct}%, "
                f"indicating a high probability of diabetic indicators."
            )
        elif d >= 0.4:
            summary_parts.append(
                f"Your diabetes risk is moderately elevated at {d_pct}%, "
                f"suggesting some pre-diabetic indicators that warrant attention."
            )
        else:
            summary_parts.append(
                f"Your diabetes risk is relatively low at {d_pct}%, "
                f"which is a positive indicator."
            )

        # Heart assessment
        if h >= 0.7:
            summary_parts.append(
                f"Your heart disease risk is concerning at {h_pct}%. "
                f"Immediate cardiological consultation is strongly recommended."
            )
        elif h >= 0.4:
            summary_parts.append(
                f"Your heart disease risk is at {h_pct}%, indicating moderate "
                f"cardiovascular concerns that should be monitored."
            )
        else:
            summary_parts.append(
                f"Your heart disease risk is at {h_pct}%, which is within "
                f"acceptable ranges."
            )

        # Mental health assessment
        if m >= 0.5:
            summary_parts.append(
                f"Our analysis of your mental health indicates signs of "
                f"depression or elevated stress (confidence: {m_pct}%). "
                f"This may also be contributing to elevated physical health risks."
            )
        else:
            summary_parts.append(
                f"Your mental health assessment shows no significant concerns "
                f"(risk level: {m_pct}%), which is excellent for overall well-being."
            )

        # Closing note
        summary_parts.append(
            "Please note that this is an AI-generated assessment and should not "
            "replace professional medical advice. Consult with qualified healthcare "
            "providers for a definitive diagnosis and treatment plan."
        )

        return " ".join(summary_parts)

    def _generate_recommendations(
        self, d: float, h: float, m: float
    ) -> list[str]:
        """
        Generate actionable health recommendations based on risk levels.

        Args:
            d: Adjusted diabetes risk probability.
            h: Adjusted heart risk probability.
            m: Mental health risk probability.

        Returns:
            List of 3-5 actionable recommendation strings.
        """
        recommendations = []

        # Diabetes-specific recommendations
        if d > 0.5:
            recommendations.append(
                "🩺 Schedule a fasting blood glucose and HbA1c test with your doctor. "
                "Monitor carbohydrate intake and consider a low-glycemic diet plan."
            )
            recommendations.append(
                "🍎 Adopt a balanced diet rich in fiber, whole grains, and lean proteins. "
                "Limit sugary beverages and processed foods."
            )
        elif d > 0.3:
            recommendations.append(
                "🍎 Consider monitoring your blood sugar levels regularly and "
                "reducing refined sugar intake as a preventive measure."
            )

        # Heart-specific recommendations
        if h > 0.5:
            recommendations.append(
                "❤️ Consult a cardiologist for a comprehensive cardiovascular evaluation. "
                "Request an ECG and lipid panel if not recently done."
            )
            recommendations.append(
                "🏃 Engage in at least 150 minutes of moderate aerobic exercise weekly. "
                "Consider activities like brisk walking, swimming, or cycling."
            )
        elif h > 0.3:
            recommendations.append(
                "❤️ Monitor your blood pressure and cholesterol levels regularly. "
                "Incorporate heart-healthy foods like nuts, fish, and leafy greens."
            )

        # Mental health recommendations
        if m > 0.5:
            recommendations.append(
                "🧠 Consider speaking with a mental health professional. Cognitive "
                "Behavioral Therapy (CBT) has proven effective for managing depression and stress."
            )
            recommendations.append(
                "🧘 Practice stress-reduction techniques such as meditation, deep breathing, "
                "or mindfulness exercises for at least 10 minutes daily."
            )
        elif m > 0.3:
            recommendations.append(
                "🧘 Maintain good mental hygiene with regular exercise, adequate sleep, "
                "and social connections to support emotional well-being."
            )

        # General lifestyle tip (always included)
        recommendations.append(
            "💧 Stay hydrated, aim for 7-9 hours of quality sleep nightly, and maintain "
            "regular physical activity for comprehensive health optimization."
        )

        # Ensure minimum 3 recommendations
        while len(recommendations) < 3:
            recommendations.insert(-1,
                "📋 Schedule a comprehensive annual health checkup with your primary "
                "care physician to establish baseline health metrics."
            )

        return recommendations
