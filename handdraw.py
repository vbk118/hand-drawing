import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

drawing_canvas = None
prev_x, prev_y = -1, -1
drawing = True  


lower_color = np.array([0, 120, 70])
upper_color = np.array([10, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if drawing_canvas is None:
        drawing_canvas = np.zeros_like(frame)

    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
       
        cnt = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y = int(x), int(y)

        if radius > 5:  
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1) 

            if drawing and prev_x >= 0 and prev_y >= 0:
                cv2.line(drawing_canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 4)

            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = -1, -1

    
    combined = cv2.addWeighted(frame, 1, drawing_canvas, 1, 0)

    cv2.imshow("Finger Drawing", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):   
        drawing_canvas = np.zeros_like(frame)
    elif key == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
