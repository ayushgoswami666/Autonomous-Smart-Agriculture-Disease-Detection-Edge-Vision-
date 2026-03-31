# API Reference

**Base URL**: `http://localhost:5000`

---

## POST `/api/image/upload`

Upload a crop leaf image for disease detection.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| image | file | Yes | JPG or PNG leaf image |

**Response `201`**
```json
{
  "imageId": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "550e8400-e29b-41d4-a716-446655440000.jpg"
}
```

**Response `400`**
```json
{ "error": "Invalid file type. Allowed: png, jpg, jpeg" }
```

---

## POST `/api/detect`

Detect crop disease from an uploaded image.

**Request** — `application/json`
```json
{ "imageId": "550e8400-e29b-41d4-a716-446655440000" }
```

**Response `200`**
```json
{
  "disease":    "Tomato___Early_blight",
  "confidence": 0.94,
  "severity":   "high",
  "top5": [
    ["Tomato___Early_blight", 0.94],
    ["Tomato___Late_blight",  0.03],
    ["Tomato___healthy",      0.02],
    ["Tomato___Leaf_Mold",    0.01],
    ["Potato___Early_blight", 0.00]
  ]
}
```

**Response `400`** — Missing imageId  
**Response `404`** — Image not found (upload first)  
**Response `503`** — Model not loaded (run training first)

---

## GET `/api/recommendation`

Get treatment recommendation for a detected disease.

**Query Parameters**

| Param | Type | Required | Description |
|---|---|---|---|
| disease | string | Yes | Disease label from `/api/detect` |

**Example**
```
GET /api/recommendation?disease=Tomato___Early_blight
```

**Response `200`**
```json
{
  "disease":   "Tomato Early Blight",
  "cause":     "Fungal — Alternaria solani",
  "symptoms":  "Dark brown spots with concentric rings on older leaves",
  "treatment": [
    "Remove and destroy infected leaves immediately",
    "Apply copper-based fungicide (Bordeaux mixture) every 7–10 days"
  ],
  "prevention": [
    "Crop rotation with non-solanaceous crops",
    "Use disease-resistant tomato varieties"
  ],
  "severity_guide": {
    "low":    "Monitor",
    "medium": "Apply fungicide",
    "high":   "Remove infected plants"
  }
}
```

**Response `400`** — Missing disease param  
**Response `404`** — No recommendation found

---

## GET `/api/history`

Retrieve all past detection records.

**Response `200`**
```json
{
  "records": [
    {
      "id":         "uuid",
      "imageId":    "uuid",
      "disease":    "Tomato___Early_blight",
      "confidence": 0.94,
      "severity":   "high",
      "timestamp":  "2026-03-15T10:30:00"
    }
  ]
}
```

---

## GET `/health`

Check API and model status.

**Response `200`**
```json
{ "status": "ok", "model_loaded": true }
```
