import cv2
import imutils

# Inisialisasi HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)

# Membaca video
cap = cv2.VideoCapture("vid.mp4")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = imutils.resize(
        frame,
        width=min(300, frame.shape[1])
    )

    # Deteksi manusia
    (regions, _) = hog.detectMultiScale(
        frame,
        winStride=(4, 4),
        padding=(4, 4),
        scale=1.05
    )

    # Gambar kotak
    for (x, y, w, h) in regions:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Jumlah pejalan kaki
    cv2.putText(
        frame,
        f"Pedestrians: {len(regions)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "Pedestrian Detection Video",
        frame
    )

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()