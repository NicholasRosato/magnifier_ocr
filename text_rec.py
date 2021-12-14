# code written by Nicholas Rosato
# sourced from https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/
# OCR recognition program for the magnifier project
# text_rec.py
# last updated 11/30/2021

import cv2
import pytesseract

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'


# Read image from which text needs to be extracted
img_name = "sample3.jpg"
img = cv2.imread(img_name)
print("Performing OCR on " + img_name)

# Preprocessing the image starts

# Convert the image to gray scale
print("Converting the image to gray scale...")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Performing OTSU threshold
print("Perfoming OTSU threshold...")
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# Applying dilation on the threshold image
print("Applying dilation on the threshold image...")
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

# Finding contours
print("Finding contours...")
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

# Creating a copy of image
print("Creating a copy of image...")
im2 = img.copy()

# A text file is created and flushed
print("Creating text file of recognized text...")
file = open("recognized.txt", "w+")
file.write("")
file.close()

# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
print("Looping through the identified contours and extracting text using pytesseract...")
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Cropping the text block for giving input to OCR
    cropped = im2[y:y + h, x:x + w]

    # Open the file in append mode
    file = open("recognized.txt", "a", encoding="utf-8")

    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped)

    # Appending the text into file
    file.write(text)
    file.write("\n")

    # Close the file
    file.close()
