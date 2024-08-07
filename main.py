import cv2
import numpy as np
import os
import pytesseract

THRESHOULD_MIN_BRIGHTNESS = 80

THRESHOULD_MAX_BRIGHTNESS = 200

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def get_image_path(document_filename):
    """Get the full path of the image file."""
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, document_filename)


def is_empty_or_null(self):
    return not self or self.strip() == ''


class DocumentImageProcessor:
    def __init__(self, document_filename):
        self.image_path = get_image_path(document_filename)
        self.image = self.load_image()

    def load_image(self):
        """Load the image from the specified path."""
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError("Image not found or unable to load.")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def check_image_quality(self):
        """Check the quality of the image."""
        if self.image is None:
            return False

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Check if the image is too bright or too dark
        brightness = np.mean(gray_image)
        if brightness > THRESHOULD_MAX_BRIGHTNESS:
            print("Imagem é muito clara.")
            return False
        elif brightness <= THRESHOULD_MIN_BRIGHTNESS:
            print("Imagem é muito escura.")
            return False

        # Check if the image is almost completely white or black
        if np.all(gray_image == 255):
            print("Imagem é completamente branca.")
            return False
        elif np.all(gray_image == 0):
            print("Imagem é completamente preta.")
            return False

        # Check if the image has any characters
        edges = cv2.Canny(gray_image, 110, THRESHOULD_MAX_BRIGHTNESS)
        if np.sum(edges) == 0:
            print("Caracteres não detectaveis na imagem.")
            return False

        text = self.read_characters()
        print("Texto extraído:\n")
        print(text)

        if is_empty_or_null(text) and THRESHOULD_MIN_BRIGHTNESS >= brightness > THRESHOULD_MAX_BRIGHTNESS:
            return False

        return True

    def is_binary_or_bilevel(self):
        """Check if the image has only one or two color tones."""
        if self.image is None:
            return False

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Calculate the histogram
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        # Find the number of non-zero bins in the histogram
        non_zero_bins = np.count_nonzero(hist)

        # If there are only 1 or 2 non-zero bins, the image is binary or bilevel
        return non_zero_bins <= 2

    def read_characters(self):
        """Read characters from the image using Tesseract OCR."""
        if self.image is None:
            return ""

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_image, lang='por')
        return text


if __name__ == "__main__":
    image_filename = "C:/dev/git/backend/python/ocr-validation/img/2.jpeg"
    processor = DocumentImageProcessor(image_filename)

    if processor.is_binary_or_bilevel():
        print("A imagem tem apenas um ou dois tons de cores.")

    if processor.check_image_quality():
        print("Qualidade da imagem aceitável.")

    else:
        print("Qualidade da imagem não aceitável.")
