import cv2
import pytesseract
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
KEYBOARD_PICTURE_PATH = ROOT_DIR + '/keyboard_picture.jpg'

def resize_with_aspect_ratio(image, target_file_size_kb, save_path, max_attempts=10):
    original_height, original_width = image.shape[:2]

    scale_factor = 1.0

    for attempt in range(max_attempts):
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        temp_save_path = f"{save_path}_temp.jpg"
        cv2.imwrite(temp_save_path, resized_image)

        file_size_kb = os.path.getsize(temp_save_path) / 1024

        print(f"Attempt {
              attempt+1}: Image size = {file_size_kb:.2f} KB at {new_width}x{new_height}")

        if file_size_kb <= target_file_size_kb:
            os.rename(temp_save_path, save_path)
            print(f"Success: Image resized to {new_width}x{
                  new_height} and saved at {file_size_kb:.2f} KB")
            return


    print("Warning: Could not reduce image to target file size within the maximum number of attempts.")


image = cv2.imread(KEYBOARD_PICTURE_PATH)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

TARGET_FILE_SIZE_KB = 100

save_path = ROOT_DIR + '/keyboard_picture_gray_resized.jpg'

resize_with_aspect_ratio(gray, TARGET_FILE_SIZE_KB, save_path)

_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

edges = cv2.Canny(thresh, 100, 200)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    if w > 150 and h > 150 and w < 850 and h < 350:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        xTextCoords = x + w // 2 - 50
        yTextCoords = y + h // 2 + 10
        
        roi = gray[y:y+h, x:x+w]
        
        config = '--psm 8'
        text = pytesseract.image_to_string(roi, config=config).strip()
        
        print(f"Detected key: {text}")
        
        cv2.putText(image, text, (xTextCoords, yTextCoords), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

cv2.imshow("Detected Keys", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
