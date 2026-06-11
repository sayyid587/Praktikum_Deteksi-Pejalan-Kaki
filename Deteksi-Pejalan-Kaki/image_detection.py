import cv2
import imutils

# Inisialisasi HOG Descriptor
hog = cv2.HOGDescriptor()

# Load model pedestrian detector bawaan OpenCV
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membaca gambar
image = cv2.imread("img.jpg")

# Resize gambar
image = imutils.resize(
    image,
    width=min(800, image.shape[1])
)

# Deteksi pejalan kaki
(regions, _) = hog.detectMultiScale(
    image,
    winStride=(8, 8),
    padding=(8, 8),
    scale=1.05
)

# Gambar kotak deteksi
for (x, y, w, h) in regions:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

# Tampilkan jumlah objek
cv2.putText(
    image,
    f"Pedestrians: {len(regions)}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

cv2.imshow("Pedestrian Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()