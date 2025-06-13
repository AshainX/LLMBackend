from PIL import Image
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ashut\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    try:
        # Preprocess image
        image = Image.open(image_path).convert('L')
        text = pytesseract.image_to_string(image, config='--psm 6')
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = [] #for diagram text splits
        for line in text.split('\n'):
            line = line.strip()
            if len(line) > 3 and not re.match(r'^[A-Z0-9]{1,2}$', line):
                lines.append(line)
        
        return '\n'.join(lines).strip()
    
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return "Could not extract question from image."
