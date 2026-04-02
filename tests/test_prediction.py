from PIL import Image
import numpy as np

# Test 1 - Check image is resized to correct size
def test_image_resize(tmp_path):
    img = Image.new("RGB", (400, 300), color=(100, 180, 60))
    img_path = str(tmp_path / "leaf.jpg")
    img.save(img_path)
    resized = Image.open(img_path).resize((224, 224))
    assert resized.size == (224, 224)
    print("Image resize test passed")

# Test 2 - Check pixel values are between 0 and 1
def test_pixel_normalization(tmp_path):
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    img_path = str(tmp_path / "white.jpg")
    img.save(img_path)
    arr = np.array(Image.open(img_path)) / 255.0
    assert arr.max() <= 1.0
    assert arr.min() >= 0.0
    print("Pixel normalization test passed")

# Test 3 - Check severity levels are correct
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
    print("Severity level test passed")
