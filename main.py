import cv2
import numpy as np
import cvzone

# Use webcam (True) or a static image (False)
USE_WEBCAM = True
IMAGE_PATH = 'path/to/your/image.jpg'  # Only used if USE_WEBCAM is False

# A4 paper dimensions in millimeters
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297

# Scale factor for the warped image. A larger scale gives more pixels per mm.
# This means warping the 210mm width to be 210 * 3 = 630 pixels.
PIXEL_PER_MM_SCALE = 3
# PIXEL_PER_MM_SCALE = 1

# Calculate the pixel dimensions of the warped A4 paper
WARPED_WIDTH = A4_WIDTH_MM * PIXEL_PER_MM_SCALE
WARPED_HEIGHT = A4_HEIGHT_MM * PIXEL_PER_MM_SCALE

cap = cv2.VideoCapture(1)  # Use 0 or 1 depending on your camera index
cap.set(3, 1280)  # camera width
cap.set(4, 720)  # camera height
cap.set(10, 160)  # brightness
cv2.namedWindow("Object Measurement", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Measurement", 900, 600)

detection_mode = True
paper_corners = None


def reorder_points(points):
    """
    Reorders 4 points to be in top-left, top-right, bottom-left, bottom-right order.
    """
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)

    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]  # Top-left has the smallest sum
    new_points[3] = points[np.argmax(add)]  # Bottom-right has the largest sum

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]  # Top-right has the smallest difference
    new_points[2] = points[np.argmax(diff)]  # Bottom-left has the largest difference

    return new_points


def find_paper_corners(img):
    """
    Finds the corners of the largest quadrilateral (assumed to be the A4 paper).
    """
    # preprocessing for contour detection
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 150)

    # Use cvzone to find contours, filtering for quadrilaterals (4 corners of the paper)
    img_contours, con_found = cvzone.findContours(img, img_canny, minArea=50000, filter=[4])

    if con_found:
        largest_contour = con_found[0]['cnt']
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        return reorder_points(approx)

    return None


while True:

    if USE_WEBCAM:
        success, img = cap.read()

        if not success:
            print("Failed to grab frame from camera.")
            break
    else:
        img = cv2.imread(IMAGE_PATH)

    img_display = img.copy()
    imgOverlay = np.zeros_like(img, dtype=np.uint8)

    if detection_mode:
        # Detect the a4 Paper
        corners = find_paper_corners(img)
        if corners is not None:
            paper_corners = corners
            cv2.drawContours(img_display, [paper_corners], -1, (0, 255, 0), 5)
            cvzone.putTextRect(img_display, "Paper Detected! Press 'c' to Confirm.",
                               (50, 50), scale=2, thickness=2, colorR=(0, 255, 0))
        else:
            cvzone.putTextRect(img_display, "A4 Paper Not Detected",
                               (50, 50), scale=2, thickness=2, colorR=(0, 0, 255))

    else:
        # Measure Objects
        # warp the image using the confirmed paper corners
        pts1 = np.float32(paper_corners)
        pts2 = np.float32([[0, 0], [WARPED_WIDTH, 0], [0, WARPED_HEIGHT], [WARPED_WIDTH, WARPED_HEIGHT]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warped = cv2.warpPerspective(img, matrix, (WARPED_WIDTH, WARPED_HEIGHT))

        img_display = img_warped.copy()

        imgOverlay = np.zeros_like(img_warped, dtype=np.uint8)

        # Find objects to measure on the warped paper
        img_warped_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
        img_warped_blur = cv2.GaussianBlur(img_warped_gray, (5, 5), 1)

        img_warped_canny = cv2.Canny(img_warped_blur, 50, 50)

        kernel = np.ones((3, 3))
        img_warped_dialated = cv2.dilate(img_warped_canny, kernel, iterations=3)
        img_warped_thresh = cv2.erode(img_warped_dialated, kernel, iterations=2)

        # Find all contours, with no filter on the number of points
        _, objects_found = cvzone.findContours(img_warped, img_warped_thresh, minArea=1300)

        if objects_found:
            for obj in objects_found:
                # Use minAreaRect for oriented bounding box
                cv2.fillPoly(imgOverlay, [np.array(obj['cnt'])], (255, 255, 0))
                rect = cv2.minAreaRect(obj['cnt'])
                box_points = cv2.boxPoints(rect)
                box_points = np.intp(box_points)

                print(box_points)

                cv2.drawContours(img_display, [box_points], 0, (0, 255, 0), 2)

                # Get width and height from the rect and convert to cm
                (w_pixels, h_pixels) = rect[1]

                # Convert pixel dimensions to cm
                width_cm = round((w_pixels / PIXEL_PER_MM_SCALE) / 10, 1)
                height_cm = round((h_pixels / PIXEL_PER_MM_SCALE) / 10, 1)

                p1, p2, p3, _ = box_points

                cv2.arrowedLine(img_display, tuple(p1), tuple(p2), (255, 0, 255), 3, 8, 0, 0.1)
                midpoint1 = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cvzone.putTextRect(img_display, f'{height_cm} cm', (midpoint1[0] - 25, midpoint1[1] - 25),
                                   scale=1.5, thickness=2, colorR=(255, 0, 0), offset=5)

                cv2.arrowedLine(img_display, tuple(p2), tuple(p3), (255, 0, 255), 3, 8, 0, 0.1)
                midpoint2 = ((p2[0] + p3[0]) // 2, (p2[1] + p3[1]) // 2)
                cvzone.putTextRect(img_display, f'{width_cm} cm', (midpoint2[0] + 15, midpoint2[1]),
                                   scale=1.5, thickness=2, colorR=(255, 0, 0), offset=5)

            # display the measurements
            # cvzone.putTextRect(img_display, f'{width_cm} cm', (box_points[1][0], box_points[1][1] - 20),
            #                    scale=1.5, colorR=(255, 0, 255))
            # cvzone.putTextRect(img_display, f'{height_cm} cm', (box_points[0][0] - 80, box_points[0][1]),
            #                    scale=1.5, colorR=(255, 0, 255))

    # img_display_resized = cv2.resize(img_display, (0, 0), None, 0.9, 0.9)
    imgStacked = cvzone.stackImages([img_display, imgOverlay], 2, 1)

    cv2.imshow("Object Measurement", imgStacked)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and paper_corners is not None:
        detection_mode = False
        print("Paper confirmed. Switched to Measurement Mode.")

    if key == ord('r'):
        detection_mode = True
        paper_corners = None
        print("Reset. Switched to Detection Mode.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
