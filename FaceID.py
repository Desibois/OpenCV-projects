import cv2
from deepface import DeepFace as DF

# Load the reference image for face recognition
reference_image = cv2.imread("Assets/Reference --Harjas.png")

# Start capturing video from the default camera (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to match the reference image size
    frame = cv2.resize(frame, (reference_image.shape[3], reference_image.shape[4]))
    # Perform face verification
    try:
        result = DF.verify(frame, reference_image, model_name='Facenet', distance_metric='euclidean_l2')
        verified = result['verified']
    except ValueError:
        verified = False

    # Display the result on the frame
    if verified:
        cv2.putText(frame, "MATCH!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    else:
        cv2.putText(frame, "NO MATCH!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Display the frame with face verification
    cv2.imshow('Facial Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
