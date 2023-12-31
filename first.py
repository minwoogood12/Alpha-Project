import cv2
import numpy as np
import pytesseract

img = './data/video.mp4' ## '영상 주소'
cap = cv2.VideoCapture(img)

def selectWord(img):
    org = img

    gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)  # ================  1 gray scale로 변환

    kernel = np.ones((2,2), np.uint8)
    kernel2 = np.ones((5,5), np.uint8)
    roi_list = []

    threshold1 = 350
    threshold2 = 100

    canny = cv2.Canny(gray, threshold1, threshold2) # ================ 2 경계 찾기

    morph = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)  # 2 ================ 경계 찾기 2

    thr = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY_INV, 3, 1)  # 3 ================ 임계처리

    
    morph2 = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel2)  # 4 ================ 뭉게기

    roi_list = []
    contours, _ = cv2.findContours(morph2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 5 ================ 특징점 찾기

    org2 = cv2.copyMakeBorder(org, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    
    

    for cnt in contours:
        try:
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(angle) > 80 and abs(angle) < 100 and 160> w >100:  # 각도가 80도에서 100도 사이인 경우 (가로 세로가 수직에 가까운 경우)
                
                roi = org2[y:y + h, x:x + w]
                roi_list.append(roi)
                cv2.rectangle(org, (x, y), (x + w, y + h), (255, 0, 0), 2)
                custom_config = r'--psm 6 --oem 3 -l kor+eng' # ================ tesseract ocr 사용
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # ================ tesseract 인식위한 이미지 흑백 전처리
                blurred = cv2.GaussianBlur(gray, (3,3), 0) # ================ tesseract 인식위한 이미지 블러 처리
                
                ''' 테서렉트 ocr 사용
                text = pytesseract.image_to_string(blurred, config=custom_config)  # --psm 6은 sparse text를 의미
              
                if not text:
                    print('Text extraction failed!')
                else:
                    print(f'Text in ROI: {text}')
                '''    
                
        
        except Exception as e:
            pass
    
    
    return org

frame_rate = cap.get(cv2.CAP_PROP_FPS) # ================ 프레임 속도 측정

while True:
    retval, frame = cap.read()

    if not retval:
        break

    frame2 = selectWord(frame)
    key = cv2.waitKey(int(1000/frame_rate))    

    cv2.imshow('frame', frame2)

    if key == 27:
        break

cv2.destroyAllWindows()
