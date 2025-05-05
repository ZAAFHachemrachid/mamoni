import unittest
from unittest.mock import Mock, patch
import customtkinter as ctk
from PIL import Image
from src.ui.preview import ImagePreviewFrame, FINAL_SIZE

class TestImagePreviewFrame(unittest.TestCase):
    def setUp(self):
        self.root = ctk.CTk()
        self.preview_frame = ImagePreviewFrame(self.root)

    def tearDown(self):
        self.root.destroy()

    def test_clear_method_resets_preview(self):
        """Test that clear() properly resets the preview state"""
        # Create a test image
        test_image = Image.new("L", (100, 100), "black")
        
        # Display test image
        self.preview_frame.display_image(test_image)
        initial_photo = self.preview_frame.photo_image
        
        # Clear the preview
        self.preview_frame.clear()
        
        # Verify a new blank image was created
        self.assertIsNotNone(self.preview_frame.photo_image)
        self.assertNotEqual(initial_photo, self.preview_frame.photo_image)
        
        # Verify image dimensions
        self.assertEqual(
            self.preview_frame.photo_image.size,
            (FINAL_SIZE, FINAL_SIZE)
        )

    def test_display_image_with_none(self):
        """Test display_image creates blank image when passed None"""
        self.preview_frame.display_image(None)
        
        self.assertIsNotNone(self.preview_frame.photo_image)
        self.assertEqual(
            self.preview_frame.photo_image.size,
            (FINAL_SIZE, FINAL_SIZE)
        )

    def test_display_image_resizes(self):
        """Test that images are properly resized"""
        test_image = Image.new("L", (200, 200), "black")
        self.preview_frame.display_image(test_image)
        
        self.assertEqual(
            self.preview_frame.photo_image.size,
            (FINAL_SIZE, FINAL_SIZE)
        )

if __name__ == '__main__':
    unittest.main()