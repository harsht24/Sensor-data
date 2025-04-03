import cv2

for i in range(5):
    print(f"\n Trying camera index {i}...")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera at index {i} is working.")
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        else:
            print(f"Camera at index {i} opened but returned no frame.")
    else:
        print(f"Camera index {i} not available.")
    cap.release()
