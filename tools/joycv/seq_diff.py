import cv2

# Load the two images to be compared
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Compute the absolute difference between the two images
diff = cv2.absdiff(gray1, gray2)

# Apply a threshold to the difference image
thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours
for contour in contours:
    # If the contour is too small, ignore it
    if cv2.contourArea(contour) < 500:
        continue

    # Compute the bounding box of the contour and draw it on the original image
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the resulting image
cv2.imshow('Movement Detection', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
