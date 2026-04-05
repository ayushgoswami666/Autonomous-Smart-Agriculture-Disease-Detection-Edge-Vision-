# Bug Report Log
## Team Winters T66 | Ashwani Chauhan | QA and Testing

## Bug 1
- Bug ID: BUG-001
- Found by: Ashwani Chauhan
- Date: March 2026
- Description: Dashboard was not showing disease result after upload
- Steps to Reproduce: Upload image and click Check for Disease button
- Expected: Disease name should appear on screen
- Actual: Screen was blank after clicking button
- Fix Applied: Connected dashboard to backend API correctly
- Status: Fixed

## Bug 2
- Bug ID: BUG-002
- Found by: Ashwani Chauhan
- Date: March 2026
- Description: Confidence score was showing as decimal instead of percentage
- Steps to Reproduce: Upload any leaf image and check result
- Expected: Score should show as 87 percent
- Actual: Score was showing as 0.87
- Fix Applied: Multiplied confidence value by 100 before displaying
- Status: Fixed

## Bug 3
- Bug ID: BUG-003
- Found by: Ashwani Chauhan
- Date: March 2026
- Description: Treatment advice was not loading for healthy plants
- Steps to Reproduce: Upload a healthy leaf image
- Expected: Message saying plant is healthy should appear
- Actual: Page was showing error message
- Fix Applied: Added healthy plant condition to treatment database
- Status: Fixed

## Summary
- Total Bugs Found: 3
- Total Bugs Fixed: 3
- Pending Bugs: 0
- Tested by: Ashwani Chauhan
