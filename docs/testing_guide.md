# Testing Guide

## Test Structure

```
tests/
├── test_api.py      — API endpoint tests (Flask test client)
└── test_model.py    — Model preprocessing & prediction tests
```

---

## Running Tests

### All tests
```bash
pytest tests/ -v
```

### With coverage report
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Single file
```bash
pytest tests/test_api.py -v
pytest tests/test_model.py -v
```

---

## Test Coverage Targets

| Area | Target | Type |
|---|---|---|
| AI Model | ≥ 80% accuracy on validation set | Functional |
| Backend API | 70% line coverage | Unit |
| UI / Dashboard | Pass rate ≥ 95% | E2E (manual) |
| Integration | All major flows pass | System |

---

## Manual Testing Checklist (Ashwani — QA)

### Dashboard UI
- [ ] Image upload accepts JPG and PNG
- [ ] Image upload rejects invalid file types
- [ ] Disease name and confidence displayed after detection
- [ ] Confidence progress bar renders correctly
- [ ] Treatment recommendations displayed after detection
- [ ] Top 5 predictions expandable section works
- [ ] Detection history shows past records
- [ ] Severity badge shows correct colour (low/medium/high)

### API (Postman / curl)
- [ ] `POST /api/image/upload` — valid image → 201 with imageId
- [ ] `POST /api/image/upload` — no file → 400
- [ ] `POST /api/detect` — valid imageId → 200 with disease
- [ ] `POST /api/detect` — missing imageId → 400
- [ ] `GET /api/recommendation?disease=Tomato___Early_blight` → 200
- [ ] `GET /api/recommendation` — no param → 400
- [ ] `GET /api/history` → 200 with records list
- [ ] `GET /health` → 200, model_loaded status

### Edge Device (Raspberry Pi)
- [ ] TFLite model loads successfully
- [ ] Single image inference completes in ≤ 3 seconds
- [ ] Camera loop captures and classifies frames continuously
- [ ] No memory leak after 10+ consecutive inferences

---

## Performance Benchmarks

Run inference timing test:
```bash
python src/edge/inference.py \
  --model models/disease_model.tflite \
  --image data/processed/Tomato___Early_blight/sample.jpg
```

Expected output:
```
Inference : 1200–2800 ms   ← target ≤ 3000 ms on Raspberry Pi 4
```

---

## Reporting Bugs

Open a GitHub Issue with:
1. Steps to reproduce
2. Expected vs actual behaviour
3. Screenshot or error log
4. Device / OS / Python version
