import cv2

def find_available_cameras(max_ids=5):
    available = []
    for i in range(max_ids):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

print("Available cameras:", find_available_cameras())

def stream_webcam(cam_id=0, width=640, height=480):
    """
    Detects a webcam and streams images.

    Args:
        cam_id (int): camera index (0 is default webcam)
        width (int): frame width
        height (int): frame height
    """
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam with id {cam_id}")

    # Optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("Webcam streaming started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Webcam Stream", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# stream_webcam()        # default webcam
# or
stream_webcam(cam_id=2)