"""
recommendations.py — Disease Treatment Recommendation Engine
Team Winters (T-66) | Autonomous Smart Agriculture Disease Detection
"""

TREATMENT_DB = {
    # Tomato diseases
    "Tomato___Early_blight": {
        "disease":     "Tomato Early Blight",
        "cause":       "Fungal — Alternaria solani",
        "symptoms":    "Dark brown spots with concentric rings on older leaves",
        "treatment": [
            "Remove and destroy infected leaves immediately",
            "Apply copper-based fungicide (Bordeaux mixture) every 7–10 days",
            "Use chlorothalonil or mancozeb sprays",
            "Avoid overhead irrigation; water at the base",
        ],
        "prevention": [
            "Crop rotation with non-solanaceous crops",
            "Use disease-resistant tomato varieties",
            "Maintain adequate plant spacing for air circulation",
            "Mulch around plants to prevent soil splash",
        ],
        "severity_guide": {"low": "Monitor", "medium": "Apply fungicide", "high": "Remove infected plants"},
    },
    "Tomato___Late_blight": {
        "disease":     "Tomato Late Blight",
        "cause":       "Oomycete — Phytophthora infestans",
        "symptoms":    "Water-soaked lesions turning dark brown on leaves and stems",
        "treatment": [
            "Apply mancozeb or metalaxyl fungicide immediately",
            "Remove and destroy severely infected plants",
            "Use systemic fungicide (cymoxanil) for rapid spread",
        ],
        "prevention": [
            "Plant certified disease-free seeds",
            "Avoid overhead watering",
            "Apply preventative copper fungicide in high humidity",
        ],
        "severity_guide": {"low": "Monitor closely", "medium": "Immediate fungicide", "high": "Destroy infected plants"},
    },
    "Tomato___Leaf_Mold": {
        "disease":     "Tomato Leaf Mold",
        "cause":       "Fungal — Passalora fulva",
        "symptoms":    "Yellow patches on upper leaf surface, olive-green mold below",
        "treatment": [
            "Apply fungicides: chlorothalonil, mancozeb, or copper compounds",
            "Improve ventilation in greenhouse or field",
            "Remove severely infected leaves",
        ],
        "prevention": [
            "Plant resistant varieties",
            "Reduce humidity with better air circulation",
            "Avoid wetting foliage during irrigation",
        ],
        "severity_guide": {"low": "Improve ventilation", "medium": "Apply fungicide", "high": "Remove affected area"},
    },
    "Tomato___healthy": {
        "disease":     "Healthy Plant",
        "cause":       "No disease detected",
        "symptoms":    "None — plant appears healthy",
        "treatment":   ["No treatment required. Continue regular care."],
        "prevention": [
            "Maintain regular watering schedule",
            "Apply balanced fertilizer",
            "Monitor regularly for early signs of disease",
        ],
        "severity_guide": {"low": "All good!", "medium": "All good!", "high": "All good!"},
    },

    # Potato diseases
    "Potato___Early_blight": {
        "disease":     "Potato Early Blight",
        "cause":       "Fungal — Alternaria solani",
        "symptoms":    "Brown spots with dark rings on lower/older leaves",
        "treatment": [
            "Apply mancozeb or chlorothalonil fungicide",
            "Remove infected leaves and tubers",
            "Ensure adequate potassium nutrition",
        ],
        "prevention": [
            "Use certified seed potatoes",
            "Crop rotation every 2–3 years",
            "Apply fungicide preventatively in wet weather",
        ],
        "severity_guide": {"low": "Monitor", "medium": "Fungicide spray", "high": "Immediate action"},
    },
    "Potato___Late_blight": {
        "disease":     "Potato Late Blight",
        "cause":       "Oomycete — Phytophthora infestans",
        "symptoms":    "Dark, water-soaked lesions; white fungal growth on underside",
        "treatment": [
            "Apply metalaxyl + mancozeb or cymoxanil fungicide",
            "Destroy all infected plant material",
            "Do not store infected tubers",
        ],
        "prevention": [
            "Use blight-resistant varieties",
            "Hill soil around plants to protect tubers",
            "Avoid overhead irrigation",
        ],
        "severity_guide": {"low": "Monitor", "medium": "Apply systemic fungicide", "high": "Harvest early if possible"},
    },

    # Corn diseases
    "Corn_(maize)___Common_rust_": {
        "disease":     "Corn Common Rust",
        "cause":       "Fungal — Puccinia sorghi",
        "symptoms":    "Small, oval, brick-red pustules on both leaf surfaces",
        "treatment": [
            "Apply triazole fungicide (propiconazole) at early rust detection",
            "Apply azoxystrobin for moderate-severe infections",
        ],
        "prevention": [
            "Plant rust-resistant hybrid varieties",
            "Early planting to avoid peak rust season",
            "Monitor fields regularly",
        ],
        "severity_guide": {"low": "Monitor", "medium": "Apply fungicide", "high": "Immediate fungicide"},
    },
}


def get_recommendation(disease: str) -> dict | None:
    """
    Return treatment recommendation for a given disease label.
    Performs partial match if exact match not found.
    """
    # Exact match
    if disease in TREATMENT_DB:
        return TREATMENT_DB[disease]

    # Partial match (case-insensitive)
    disease_lower = disease.lower()
    for key, value in TREATMENT_DB.items():
        if disease_lower in key.lower() or key.lower() in disease_lower:
            return value

    # Healthy fallback
    if "healthy" in disease_lower:
        return TREATMENT_DB.get("Tomato___healthy")

    return None
