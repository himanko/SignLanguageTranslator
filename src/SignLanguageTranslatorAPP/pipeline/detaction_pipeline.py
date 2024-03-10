import cv2
from SignLanguageTranslatorAPP.components.processing import PreprocessFrame
from SignLanguageTranslatorAPP.utils.preprocessing_utils import FeaturePreprocess

class DetectionPipeline:
    def __init__(self) -> None:
        # Initialize landmark feature preprocessing
        self.feature_preprocess = FeaturePreprocess()

    def detection(self):
        # Open webcam
        cap = cv2.VideoCapture(0)

        # Check if webcam opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()

            # Check if frame is empty
            if not ret:
                print("Error: Empty frame.")
                break

            # Preprocess the frame
            preprocess_frame = PreprocessFrame()
            preprocessed_frame = preprocess_frame.preprocess_frame(frame)

            # Display the preprocessed frame
            cv2.imshow('Real-time Detection', preprocessed_frame)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release webcam and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

# Instantiate and start the detection pipeline
if __name__ == "__main__":
    detection_pipeline = DetectionPipeline()
    detection_pipeline.detection()
