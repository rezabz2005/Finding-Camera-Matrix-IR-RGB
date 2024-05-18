# Returns The FindingCorners of The Input Picture

import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

obj_points = []
img_points = []

# Give The Name of Input Image You Want To Work With

images = glob.glob("calib_result.jpg")

for frame in images:
    img = cv2.imread(frame)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    if ret == True:
        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        img_points.append(corners2)

        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow("image",img)

        # This Piece of Code Below Will Save The Image Where You Want
        # cv2.imwrite("Whatever Saving Path THat You Want", img)

        key = cv2.waitKey(0)
        if key == ord("Q"):
            break

# It Gives You The Wanted FindingCorners In Selected Picture

cv2.destroyAllWindows()

# By Using The Ending of The First Code You Can Extrat The Camera and Transformation Matrix