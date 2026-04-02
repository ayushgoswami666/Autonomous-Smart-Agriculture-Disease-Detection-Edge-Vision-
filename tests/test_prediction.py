from PIL import Image
import numpy as np

def test_image_resize(tmp_path):
    img = Image.new("RGB", (400, 300), color=(100, 180, 60))
    img_path = str(tmp_path / "leaf.jpg")
    img.save(img_path)
    resized = Image.open(img_path).resize((224, 224))
    assert resized.size == (224, 224)

def test_severity_levels():
    def get_severity(confidence):
        if confidence >= 0.85:
            return "high"
        elif confidence >= 0.60:
            return "medium"
        else:
            return "low"
    assert get_severity(0.90) == "high"
    assert get_severity(0.70) == "medium"
    assert get_severity(0.40) == "low"
