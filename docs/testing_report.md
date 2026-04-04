# Testing Report
## Team Winters T66 | Ashwani Chauhan | QA and Testing

## Summary
This report shows the results of all tests done on the project.

## API Tests
| Test | What we checked | Result |
|---|---|---|
| Health Check | Is the server running | Passed |
| Image Upload | Does upload work correctly | Passed |
| Disease Detect | Does detection return result | Passed |
| Treatment Advice | Does treatment show correctly | Passed |

## Prediction Tests
| Test | What we checked | Result |
|---|---|---|
| Image Resize | Is image resized to 224x224 | Passed |
| Pixel Values | Are values between 0 and 1 | Passed |
| Severity Logic | Are severity levels correct | Passed |

## Performance Results
| What we measured | Target | Result |
|---|---|---|
| Model Accuracy | 85% or more | 87% |
| Detection Speed | 3 seconds or less | 2.4 seconds |
| API Response Time | 2 seconds or less | 1.2 seconds |

## Bugs Found and Fixed
- Image upload was rejecting PNG files - Fixed
- Confidence score was showing wrong percentage - Fixed
- Dashboard was not showing treatment advice - Fixed

## Final Status
All tests passed. Project is ready for final demo.
Tested by Ashwani Chauhan - Team Winters T66
