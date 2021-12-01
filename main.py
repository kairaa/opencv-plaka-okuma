import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('images/1.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

haar_cascade = cv2.CascadeClassifier('haarcascade_plate_number.xml')


def carplate_detect(image):
    carplate_overlay = image.copy()

    carplate_rects = haar_cascade.detectMultiScale(carplate_overlay, scaleFactor=1.1, minNeighbors=3)

    for x, y, w, h in carplate_rects:
        cv2.rectangle(carplate_overlay, (x, y), (x+w, y+h), (0, 255, 0), 5)

    return carplate_overlay


def carplate_extract(image):
    carplate_rects = haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in carplate_rects:
        carplate_img = image[y + 15:y + h - 10, x + 15:x + w - 20]

    return carplate_img


def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


detected_img = carplate_detect(img_rgb)

carplate_extract_img = carplate_extract(img_rgb)
carplate_extract_img = enlarge_img(carplate_extract_img, 150)

carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)

carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray, 3)

print(pytesseract.image_to_string(carplate_extract_img_gray_blur,
                                  config=f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

cv2.imshow('img', carplate_extract_img_gray_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
