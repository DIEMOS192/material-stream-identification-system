import cv2
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference import SVMFrameClassifier


def main():
    base_dir = Path(__file__).resolve().parents[1] / "models"
    classifier = SVMFrameClassifier.from_model_directory(
        base_dir / "saved_models"
    )

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True: 
        is_valid, frame = cam.read()
        if not is_valid:
            print("Error: Could not read frame from camera.")
            return
        

        prediction = classifier.predict(frame)
        if prediction.rejected:
            result = "Prediction: Unknown Object"
        else:
            result = f"Prediction: {prediction.label} (Confidence: {prediction.confidence:.4f})"
        
        green, red = (0, 255, 0), (0, 0, 255)
        color = red if prediction.rejected else green
        cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Camera opened, press 'q' to quit", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break


    
    cam.release()
    cv2.destroyAllWindows()
    

    pass




if __name__ == "__main__":
    main()








