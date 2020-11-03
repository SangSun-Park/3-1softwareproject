import numpy as np
import cv2
import FaceDetector as models
import os

import time
from face_learn import compare


class Box:
    def __init__(self, sx, sy, ex, ey, name):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        self.w = ex - sx
        self.h = ey - sy
        self.label = name


def modify_box(box, w, h):
    if box.sx < 0:
        box.sx = 0
    if box.sy < 0:
        box.sy = 0
    if box.ex >= w:
        box.ex = w - 1
    if box.ey >= h:
        box.ey = h - 1
    box.w = box.ex - box.sx
    box.h = box.ey - box.sy

    return box


def compare_box(box1, box2):
    r = 0.85
    ratio = (1 - 0.7) / 2
    # 박스의 크기가 비슷하면서 중심점이 상대의 내부에 있을 경우
    # 서로가 일정 비율의 박스로 상대 박스 안에 있을 경우
    # 완전히 같거나 아예 들어간 경우는 0, 유사한 경우는 1, 아예 상관없으면 -1
    # 1은 복사 or 삭제 대상. 0, 2는 삭제 대상.

    i = -1
    if box1.label == "NONE" or box2.label == "NONE" or box1.label != box2.label:
        return i

    # 비슷한 위치에 유사한 박스인 경우
    if ((box1.sx + (box1.w * ratio)) > box2.sx and (box1.ex - (box1.w * ratio)) < box2.ex and
            (box1.sy + (box1.h * ratio)) > box2.sy and (box1.ey - (box1.h * ratio)) < box2.ey and
            (box2.sx + (box2.w * ratio)) > box1.sx and (box2.ex - (box2.w * ratio)) < box1.ex and
            (box2.sy + (box2.h * ratio)) > box1.sy and (box2.ey - (box2.h * ratio)) < box1.ey):
        i = 3

    # 박스가 유사한 경우
    if (box1.w > box2.w * r or box1.h > box2.h * r) and (box1.w * r < box2.w or box1.h * r < box2.h):
        i = 1

    # box1이 메인이기 때문에 삭제 대상으로 잡을것은 box1이 안에 들어간 경우.
    if box1.sx > box2.sx and box1.ex < box2.ex and box1.sy > box2.sy and box1.ex < box2.ex:
        i = 2

    # 완전히 똑같은 경우
    if box1.sx == box2.sx and box1.sy == box2.sy and box1.w == box2.w and box1.h == box2.h:
        i = 0

    return i


def combine_box(box1, box2):
    tmp_box = Box(box1.sx, box1.sy, box1.sx + box1.w, box1.sy + box1.h, box1.label)
    # 겹쳐 있는 경우
    # 하나의 꼭짓점이 다른 공간에 포함되어 있을 때 겹쳐있다고 판단
    if ((box1.sx >= box2.sx and box1.sx <= (box2.sx + box2.w) and
            ((box1.sy >= box2.sy and box1.sy <= (box2.sy + box2.h) or
              box2.sy <= (box1.sy + box1.h) <= (box2.sy + box2.h)))
        ) or
        ((box1.sx + box1.w) >= box2.sx and (box1.sx + box1.w) < (box2.sx + box2.w) and
            ((box1.sy >= box2.sy and box1.sy <= (box2.sy + box2.h) or
            ((box1.sy + box1.h) >= box2.sy and (box1.sy + box1.h) <= (box2.sy + box2.h))))
        ) or
        (box2.sx >= box1.sx and box2.sx <= (box1.sx + box1.w) and
            ((box2.sy >= box1.sy and box2.sy <= (box1.sy + box1.h) or
            (box2.sy + box2.h) >= box1.sy and (box2.sy + box2.h) <= (box1.sy + box1.h)))
        ) or
        ((box2.sx + box2.w) >= box1.sx and (box2.sx + box2.w) <= (box1.sx + box1.w) and
            ((box2.sy >= box1.sy and box2.sy <= (box1.sy + box1.h) or
            (box2.sy + box2.h) >= box1.sy and (box2.sy + box2.h) <= (box1.sy + box1.h)))
        )):

        if box1.sx > box2.sx:
            tmp_box.sx = box2.sx
            tmp_box.w = tmp_box.w + (box1.sx - box2.sx)

        if box1.sx + box1.w < box2.sx + box2.w:
            tmp_box.w = tmp_box.w + (box2.sx + box2.w) - (box1.sx + box1.w)

        if box1.sy > box2.sy:
            tmp_box.sy = box2.sy
            tmp_box.h = tmp_box.h + (box1.sy - box2.sy)

        if box1.sy + box1.h < box2.sy + box2.h:
            tmp_box.h = tmp_box.h + (box2.sy + box2.h) - (box1.sy + box1.h)

    return tmp_box


class faceDetection:
    label = []
    faces = []  # [boxes]
    frame_amount = -1

    def __init__(self, fps, names, model, in_enc, labels):
        self.frame_amount = int(fps * 2 / 3)
        self.faces.clear()
        dirPath = "data/Video/"
        video_images = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
        video_images.sort()
        empty_faces = 0
        for i, imgFile in enumerate(video_images):
            print(f"Processing Marking Box {i + 1} / {len(video_images)}")
            image = imread_utf8(dirPath + video_images[i])
            boxes = face_detect(image)
            if len(boxes) == 0:
                empty_faces += 1
            else:
                empty_faces = 0
            self.faces.append(boxes)
            self.arrange_boxes(i)

            start = time.time()
            not_remove_boxes = []

            for box in boxes:
                face = image[box.sy:box.ey, box.sx:box.ex]
                cmp = compare(model, in_enc, labels, face, names)
                if cmp is not None:
                    tmpbox = Box(box.sx, box.sy, box.ex, box.ey, cmp)
                    not_remove_boxes.append(tmpbox)

            self.faces[i].clear()
            self.faces[i].extend(not_remove_boxes)
            not_remove_boxes.clear()

            end = time.time()
            print(f"Find faces in {i + 1}: ", format(end - start, '.2f'), "s")

            if i > self.frame_amount > empty_faces:
                start = time.time()
                self.face_checking(i)
                end = time.time()
                print("Checking other box: ", format(end - start, '.2f'), "s")

    def face_checking(self, index):
        # 일정 프레임 이내의 상태를 확인하고 알맞는 박스가 있으면 같은 사람으로 인식
        # 현재 프레임 내에서 이미 확인한 박스인지 확인하는 List
        indexes = []
        # 시작점 체크
        startIndex = index - self.frame_amount
        for i, boxes in enumerate(self.faces[startIndex:index - 1]):
            # 프레임 넘버에 해당하는 박스 목록을 차례로 비교.
            for box1 in boxes:
                for k, box2 in enumerate(self.faces[index]):
                    cp = compare_box(box1, box2)
                    if (k not in indexes) and (cp == 0 or cp == 1):
                        j = i + 1 + startIndex
                        rx = (box2.ex + box2.w/2) - (box1.ex + box1.w/2)
                        ry = (box2.ey - box2.h/2) - (box1.ey - box1.h/2)
                        while j < index:
                            isSkip = False
                            # 만약에 이미 비슷한 얼굴이 있다면 박스를 해쳐서는 안되므로
                            for b in self.faces[j]:
                                ccp = compare_box(box1, b)
                                if ccp == 0 or ccp == 1:
                                    isSkip = True
                                    break
                            if isSkip is False:
                                ratio = float((j - i - startIndex) / (self.frame_amount - i))
                                sx = int(box1.sx + rx * ratio)
                                ex = int(box1.ex + rx * ratio)
                                sy = int(box1.sy + ry * ratio)
                                ey = int(box1.ey + ry * ratio)
                                tbox = Box(sx, sy, ex, ey, box1.label)
                                self.faces[j].append(tbox)
                            j += 1

        if len(indexes) > 0:
            i = startIndex
            while i < index:
                self.arrange_boxes(i)
                i += 1

    # 중복되는 box들은 지워줌
    def arrange_boxes(self, index):
        # print(f"{index}. Remove redundant boxes among {len(self.faces[index])} boxes")
        small_boxes = []
        boxes = []
        labels = []
        for i, box1 in enumerate(self.faces[index]):
            for j, box2 in enumerate(self.faces[index]):
                if (i != j) and (box1 not in boxes):
                    cp = compare_box(box1, box2)
                    # 작은 상자들은 따로 먼저 처리해줘야함.
                    if cp == 2:
                        small_boxes.append(box1)
                        break
                    elif cp != -1 and cp != 1:
                        boxes.append(box1)
                        label = 0
                        isBreak = False
                        # label을 통해 비슷한 위치끼리 묶어서 하나가 남을 수 있도록 따로 저장해줌. 후에 비교할 때 사용
                        while label < len(labels) + 1 and isBreak is False:
                            # labels에 없던 box라면 새롭게 만들어줘야함.
                            if label == len(labels):
                                labels.append([box1])
                                break
                            # labels에 이미 있던 box라면 추가해줌.
                            else:
                                for label_box in labels[label]:
                                    ccp = compare_box(box1, label_box)
                                    if ccp != -1 and ccp != 1:
                                        labels[label].append(box1)
                                        isBreak = True
                                        break
                            label += 1
                        break
        for small_box in small_boxes:
            self.faces[index].remove(small_box)

        # 중복된 box를 지울 때 다 지우면 안되므로 한개 남겨야함.
        for i, box in enumerate(boxes):
            for label_boxes in labels:
                if len(label_boxes) <= 1:
                    break
                if box in label_boxes:
                    self.faces[index].remove(box)
                    label_boxes.remove(box)
                    break


# 블러 처리
def draw_face(image, boxes):
    covered_boxes = []
    # 블러처리 과정에서 겹쳐진 박스들을 합쳐주는 과정
    for i, box in enumerate(boxes):
        for j, comp_box in enumerate(boxes):
            # 같은 인물일 경우 합쳐주는 과정 진행
            if (i != j) and box.label == comp_box.label:
                box = combine_box(box, comp_box)

        is_covered = False
        # 합쳐지면 같은 박스가 됨. 처리가 끝난 박스는 처리 X
        for cov_box in covered_boxes:
            cp = compare_box(box, cov_box)
            if cp == 0:
                is_covered = True
                break

        if is_covered is False:
            face_img = image[box.sy:+box.ey, box.sx:box.ex]
            face_img = cv2.blur(face_img, (70, 70))
            image[box.sy:box.ey, box.sx:box.ex] = face_img
            covered_boxes.append(box)
    return image


def face_detect(image):
    boxes = []

    (height, width) = image.shape[:2]

    # DNN에서 얼굴 찾기
    detections_dnn, h, w = find_faces_dnn(image)
    dnn_face_amount = 0
    for i in range(0, detections_dnn.shape[2]):
        confidence = detections_dnn[0, 0, i, 2]
        if confidence > 0.15:
            dnn_face_amount += 1
            dnn_box = detections_dnn[0, 0, i, 3:7] * np.array([w, h, w, h])
            sx, sy, ex, ey = dnn_box.astype("int")
            box = Box(sx, sy, ex, ey, "NONE")
            box = modify_box(box, width, height)
            boxes.append(box)

    # MTCNN에서 얼굴 찾기
    face_mtcnn_list = find_faces_mtcnn(image)
    if len(face_mtcnn_list) > 0:
        for face in face_mtcnn_list:
            x, y, w, h = face['box']
            box = Box(x, y, x + w, y + h, "NONE")
            box = modify_box(box, width, height)
            boxes.append(box)

    return boxes


def find_faces_mtcnn(img):
    start = time.time()
    faces = models.mtcnn_detector.detect_faces(img)
    end = time.time()
    print("MTCNN : ", format(end - start, '.2f'))
    return faces


def find_faces_dnn(img):
    start = time.time()
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    models.dnn_detector.setInput(blob)
    detections = models.dnn_detector.forward()
    end = time.time()
    print("DNN : ", format(end - start, '.2f'))
    return detections, h, w


def find_one_face_dnn(temp, filename):
    img = imread_utf8(temp)
    count = 0
    faces, h, w = find_faces_dnn(img)
    for j in range(0, faces.shape[2]):
        confidence = faces[0, 0, j, 2]
        if confidence > 0.5:
            count += 1
            if count > 1:
                break
            box = faces[0, 0, j, 3:7] * np.array([w, h, w, h])
            sx, sy, ex, ey = box.astype("int")
            faceImg = img[sy:ey, sx:ex]
            fH, fW = faceImg.shape[:2]
            if fW < 20 or fH < 20:
                count -= 1
                break

            faceImg = cv2.resize(faceImg, dsize=(150, 150), interpolation=cv2.INTER_LINEAR)

    os.remove(temp)
    if count == 1:
        print("Find one face")
        imwrite_utf8(filename, img)
        return True
    else:
        print("face is not only one")
        return False


def imread_utf8(img_path, flags=cv2.IMREAD_COLOR):
    try:
        new_imgPath = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(new_imgPath, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite_utf8(img_path, img, params=None):
    try:
        ext = os.path.splitext(img_path)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(img_path, mode='wb') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return None
