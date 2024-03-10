import cv2


# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    
    # Display the preprocessed frame
    # cv2.imshow('Real-time Detection', preprocessed_frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()