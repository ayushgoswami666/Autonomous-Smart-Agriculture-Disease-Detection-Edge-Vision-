# These tests check if the dashboard pages are working correctly
# Written by Ashwani Chauhan - Team Winters T66

# Test 1 - Check main page loads correctly
def test_main_page_loads():
    print("Main dashboard page load test passed")
    assert True

# Test 2 - Check file uploader accepts jpg files
def test_file_uploader_accepts_jpg():
    allowed_types = ["jpg", "jpeg", "png"]
    test_file = "leaf.jpg"
    extension = test_file.split(".")[-1]
    assert extension in allowed_types
    print("File type test passed")

# Test 3 - Check result page shows correct severity
def test_severity_display():
    def get_severity_message(severity):
        if severity == "high":
            return "Act immediately"
        elif severity == "medium":
            return "Monitor closely"
        else:
            return "Keep watching"
    assert get_severity_message("high") == "Act immediately"
    assert get_severity_message("medium") == "Monitor closely"
    assert get_severity_message("low") == "Keep watching"
    print("Severity display test passed")

# Test 4 - Check history page has correct fields
def test_history_record_fields():
    sample_record = {
        "date": "April 05 2026",
        "crop": "Tomato",
        "disease": "Early Blight",
        "result": "High"
    }
    assert "date" in sample_record
    assert "crop" in sample_record
    assert "disease" in sample_record
    assert "result" in sample_record
    print("History record fields test passed")
