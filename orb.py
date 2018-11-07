import cv2

# 画像ファイルの読み込み
img = cv2.imread('mark.png')

# ORB (Oriented FAST and Rotated BRIEF)
detector = cv2.ORB_create()

# 特徴検出
keypoints = detector.detect(img)

# 画像への特徴点の書き込み
out = cv2.drawKeypoints(img, keypoints, None)

# 表示
cv2.imshow("keypoints", out)