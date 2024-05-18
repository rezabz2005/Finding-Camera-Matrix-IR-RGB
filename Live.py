
# Returns The Wanted Camera Matrix
# By Opening The Camera And Taking Pictures With "S" and quitting With "Q" While Saving The Images

import cv2 as cv
import numpy as np
import os

# Give The Wanted Chess Grid Size

CHESS_BOARD_DIM = (7, 7)

n = 0

image_dir_path = "images"

CHECK_DIR = os.path.isdir(image_dir_path)

if not CHECK_DIR:
    os.makedirs(image_dir_path)
    print(f"{image_dir_path}Directory is created")
else:
    print(f"{image_dir_path}Directory already Exists.")

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret == True:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)

    return image, ret


cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    copyFrame = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    image, board_detected = detect_checker_board(frame, gray, criteria, CHESS_BOARD_DIM)

    cv.putText(
        frame,
        f"saved_img : {n}",
        (30, 40),
        cv.FONT_HERSHEY_PLAIN,
        1.4,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    cv.imshow("frame", frame)
    cv.imshow("copyFrame", copyFrame)

    key = cv.waitKey(1)

    if key == ord("q"):
        break
    if key == ord("s") and board_detected == True:

        cv.imwrite(f"{image_dir_path}/image{n}.png", copyFrame)

        print(f"saved image number {n}")
        n += 1
cap.release()
cv.destroyAllWindows()

print("Total saved Images:", n)

# Give The Length of Each Square

SQUARE_SIZE = 9

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


calib_data_path = "../calib_data"
CHECK_DIR = os.path.isdir(calib_data_path)


if not CHECK_DIR:
    os.makedirs(calib_data_path)
    print(f"{calib_data_path}Directory is created")

else:
    print(f"{calib_data_path}Directory already Exists.")

obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)

obj_3D[:, :2] = np.mgrid[0 : CHESS_BOARD_DIM[0], 0 : CHESS_BOARD_DIM[1]].T.reshape( -1, 2)
obj_3D *= SQUARE_SIZE
print(obj_3D)

obj_points_3D = []
img_points_2D = []

image_dir_path = "images"

files = os.listdir(image_dir_path)
for file in files:
    print(file)
    imagePath = os.path.join(image_dir_path, file)

    image = cv.imread(imagePath)
    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_DIM, None)
    if ret == True:
        obj_points_3D.append(obj_3D)
        corners2 = cv.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners2)

        img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None)
print("calibrated")

print("putting the data into one files using numpy ")
np.savez(
    f"{calib_data_path}/MultiMatrix",
    camMatrix=mtx,
    distCoef=dist,
    rVector=rvecs,
    tVector=tvecs,)

print("-------------------------------------------")

print("loading data stored using numpy save-z function\n \n \n")

data = np.load(f"{calib_data_path}/MultiMatrix.npz")

camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

print("loaded calibration data successfully")

npz = np.load('MultiMatrix.npz')

print(npz)
print(npz["distCoef"])
print(npz["rVector"])
print(npz["tVector"])
print(npz["camMatrix"])

# If You Provide The Matrix of Two Cameras In matrix_1 and matrix_2 It Will Return You The Transformation Matrix

# npz = np.load('MultiMatrix.npz')
# matrix_1 = npz["camMatrix"]
# matrix_2 = npz["camMatrix"]

# matrix_1 = np.linalg.inv(matrix_1)
# print(npz["camMatrix"])
# print(npz["camMatrix"])
# result = np.dot(matrix_2, matrix_1)
# print(result)
