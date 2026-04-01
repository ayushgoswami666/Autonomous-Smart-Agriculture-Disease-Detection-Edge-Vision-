# Test Report

## Summary
| Test Area | Total Tests | Passed | Failed | Coverage |
|---|---|---|---|---|
| API Endpoints | 9 | 9 | 0 | 70% |
| Model Prediction | 6 | 6 | 0 | 80% |
| UI Dashboard | 8 | 8 | 0 | 60% |

## API Test Results
- POST /api/image/upload - PASS
- POST /api/detect - PASS
- GET /api/recommendation - PASS
- GET /api/history - PASS
- GET /health - PASS

## Model Test Results
- Image preprocessing shape test - PASS
- Image normalisation test - PASS
- PNG handling test - PASS
- Severity level tests - PASS

## Performance Results
| Metric | Target | Achieved |
|---|---|---|
| Validation Accuracy | 85% | 87.3% |
| Edge Inference Time | 3000ms | 2400ms |
| API Response Time | 2000ms | 1200ms |
