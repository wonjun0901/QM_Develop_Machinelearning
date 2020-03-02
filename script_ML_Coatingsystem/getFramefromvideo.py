# 동영상 프로세싱

import cv2


vidcap = cv2.VideoCapture(
    "D:/dropbox/Dropbox/3. 연구/[개발팀]/20200228 디스펜스 코팅/zero_point.avi")


count = 0

while(vidcap.isOpened()):
    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
    ret, image = vidcap.read()

    # 캡쳐된 이미지를 저장하는 함수
    cv2.imwrite(
        "D:/DEV/Python/QM_Develop_Machinelearning/frame/frame1%d.jpg" % count, image)

    print('Saved frame%d.jpg' % count)
    count += 1

vidcap.release()
