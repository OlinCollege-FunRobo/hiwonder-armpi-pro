import cv2

#cap = cv2.VideoCapture(0)
#while True:
#    ret, frame = cap.read()
#    if not ret:
#        break

#    cv2.imshow('Camera frame', frame)
#    if cv2.waitKey(1) == ord('q'):
#        break


#cap.release()
#cv2.destroyAllWindows()


# Use to check to find where the usb camera is located
# import cv2

available_cameras = []
for device_id in range(5):
    cap = cv2.VideoCapture(device_id)
    if cap.isOpened():
        print(f"Found camera at index {device_id}")
        available_cameras.append(device_id)
        cap.release()
    else:
        print(f"No camera at index {device_id}")




