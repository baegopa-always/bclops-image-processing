

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import MeanShift, estimate_bandwidth

# hough 변환 알고리즘##############################
def hough(red_points, houghThreshold, minLength, maxGap, onImg):
    lines = cv2.HoughLinesP(red_points, 1, np.pi / 180, threshold=houghThreshold, minLineLength=minLength,
                            maxLineGap=maxGap)
    return lines


# 허프 변환으로 추출한 선분들 (lines) 의 각도 구하기 / 범위는 0도~180도##############################
def calculate_angle(start, end):
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]

    # 두 점의 좌표를 이용하여 각도를 계산합니다.
    angle = math.atan2(y2 - y1, x2 - x1)  # 라디안 단위로 계산됩니다.
    angle = math.degrees(angle)  # 각도로 변환합니다.

    # 결과값을 0 ~ 180도 범위로 조정합니다.
    if angle < 0:
        angle += 180

    return int(angle)


# 길이 구하기#####################################
def distance(start, end):
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]

    dis = (x1 - x2) ** 2 + (y1 - y2) ** 2
    dis = round(math.sqrt(dis), 2)

    return dis


# 절리 중심점 구하기##############################
def halfPoints(points, half_points, index):

    x1 = points[0]
    y1 = points[1]
    x2 = points[2]
    y2 = points[3]

    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    
    half_points.append([x,y, index])
    return 0


# 카메라에서 실제 거리 구하기##############################
class Lens:
    # 0.25, 0.4, 0.65, 1.2mm
    def __init__(self, distance=1000, fl=12, siah=6.287, srph=4050):
        self.FL = fl  # FL: focalLength(초점길이)
        self.WD = distance * 0.001  # WD: workingDistance 0.34 물체와 카메라 사이의 거리
        self.SIAH = siah  # SIAH: sensorImageArea 14.0 센서 이미지 영역
        self.SRPH = srph  # SRPH: sensorResolßßßution 9248 센서 해상도

        # if distance <= 2000 : distance += 1000

        self.PMAG = self.FL / (distance)  # PMAG
        self.HFOV = self.SIAH / self.PMAG  # HFOV
        self.SPP = self.HFOV / self.SRPH  # SPP: Size Per Pixel

        # print("SPP(0.39):", self.SPP, ",  HFOV(1572):", self.HFOV, ", PMAG(0.004):", self.PMAG)
        self.R = 0.045 * ((self.WD) ** 2) - 0.355 * (self.WD) + 0.82  # 감소계수

        # ~~~~~~~~~~~~~~~~ REMOVABLE ~~~~~~~~~~~~~~~~~~~~~~
        self.R *= 1.2

    def real_length(self, pixel_length_list):
        real_length_list = []
        # print(pixel_length_list[0] * self.SPP * self.R )
        for i in range(len(pixel_length_list)):
            real_length_list.append(pixel_length_list[i] * self.SPP * self.R)
        # real_length_list = [pixels * self.SPP * self.R * 10000 for pixels in pixel_length_list]
        return real_length_list


# 선분을 극좌표로 변경
def getpolar(x1, y1, x2, y2):
    # 선분의 기울기 계산
    if (x2 == x1):
        theta = 0
        r = abs(x1)
    else:
        m = (y2 - y1) / (x2 - x1)
        # 라디안 단위 각도 계산
        theta = (np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi) + 90
        if theta > 180: theta = theta - 360
        # r 계산
        b = y1 - m * x1
        A = -m
        B = 1
        C = -b
        r = abs(C) / np.sqrt(A ** 2 + B ** 2)
    return theta, r


red = (0, 0, 255)
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)

# red-black 이미지로 변환
def redImage(image): 
    img = image.copy()
    height = img.shape[0]
    width = img.shape[1]

    # red 좌표들만 따서 list 에 넣기
    points = []

    # red 좌표 아닌 부분은 black 으로 변환
    for i in range(height):
        for j in range(width):
            if not (np.array_equal(img[i, j], red)):
                img[i, j] = black
            else:
                img[i,j] = white
                points.append((i, j))
    pointsImg = img.copy()
    cv2.imwrite('pointsImg.jpg', pointsImg)
    return img


# 이미지에서 라인 가져와서 각도별로 clustering
def getLine(redImage) :
    red_points = redImage.copy()

    # canny 필터 적용 후 hough 알고리즘을 이용하여 후보 직선의 시작점과 끝점을 추출
    edges = cv2.Canny(red_points, 100, 500,apertureSize = 3)
    lines = hough(edges, 50, 140, 50, red_points)
    for points in lines:
        cv2.line(red_points, (points[0][0],points[0][1]), (points[0][2],points[0][3]), red, 3)
    lineImg = red_points.copy()
    cv2.imwrite('resultLine.jpg', lineImg)

    hough_angle = []
    for line in lines:
        hough_angle.append(round(calculate_angle([line[0][0], line[0][1]],[line[0][2], line[0][3]]) / 10 ))

    # 리스트 내의 값은 해당 각도의 lines의 index 값
    # clustering 한 각도들 중에서 최소 3개 이상인 각도의 index 를 이용해서 양 끝점 저장하기
    angleCnt = [ 0 for i in range(19)]
    angleLines = [[] for i in range(19)]

    i = 0
    # 같은 각도를 가진 절리의 개수 카운트
    for ind in hough_angle:
        # ind = 각 절리 각도값 / 10
        angleCnt[ind] += 1
        angleLines[ind].append(lines[i][0])
        i += 1
    print("angleCnt: ", angleCnt)
    # print("angle liines\n", angleLines)

    cluster_points = []
    
    # 1. 10 개 이상인 절리들 양 끝점 넣기
    # 2. half points 구하기 -> 정렬 후 시작 점 y 값 비교하기?
    # 3. 각도값에 따라 x or y 값 기준으로 정렬 
    # 4. 값 차이가 30 이상인 경우 new [] 생성해서 넣기
    # 5. 각 []의 count 값이 5개 미만인 경우 버리기 
    # 6. x, y 값으로 정렬해서 양 끝점 연결 가능 ? 
    finalJointSet = []
    finalInd = -1
    for i in range(19) :
        # 1. 10 개 이상인 절리들 양 끝점 넣기
        img = cv2.imread('pointsImg.jpg')
        before = 0
        
        # 같은 각도의 절리가 10개 이상인 경우 - 하나의 절리군임 집합임
        if (angleCnt[i] >= 10) :
            finalJointSet.append([])
            finalInd += 1
            newLines = angleLines[i]
            # print('newLines\n', newLines, '\n')
            
            XorY = 0
            if (i >= 5 and i <= 13):
                newLines.sort(key=lambda x:x[0])
                before = newLines[0][0]
                XorY = 0
            else :
                newLines.sort(key=lambda x:x[1])
                before = newLines[0][1]
                XorY = 1
                
            resultPoints = [[]]
            resultInd = 0
            
            
            for points in newLines:
                # print("newLine points\n", points, "\n")
                if (abs(points[XorY] - before) <= 25) :
                    resultPoints[resultInd].append(points)
                else :
                    resultPoints.append([])
                    resultInd += 1
                    resultPoints[resultInd].append(points)
                    before = points[XorY]
            # print('result points\n', resultPoints, "\n")
            resultPoints[0].sort(key=lambda x:x[0]) # 시작점 x 값 기준으로 정렬 
            img = cv2.imread('pointsImg.jpg')
            
            
            for ind, points in enumerate(resultPoints):
                
                print(len(points), "\n")
                
                if (len(points) > 2) :
                    points.sort(key=lambda x:x[0]) # 시작점 x 기준으로 정렬
                    start = (points[0][0], points[0][1])
                    # print('start', start)
                    
                    points.sort(key=lambda x:x[2], reverse=True) # 끝점 x 기준으로 정렬
                    end = (points[0][2], points[0][3])
                    # print('end', end, "\n")
                    
                    cv2.line(img, (start[0], start[1]), (end[0], end[1]), green, 3) # 해당 절리군에 포함되는 최종 절리
                    
                    finalJointSet[finalInd].append([start[0], start[1], end[0], end[1]])
            # print('final joint sets\n', finalJointSet[finalInd])
            url = 'resultPoints' + str(ind) + '.jpg'
            cv2.imwrite(url, img)       
    return finalJointSet
        
# joint 길이 구하기
def joint_length(joint_points, lens) :
    print('length joint_points: ', joint_points)

    jointset_length_list = []  
    for i, jointset in enumerate(joint_points):#jointset 별로 길이를 다른 배열에 넣게 수정하였습니다
        joint_length_list = []
        print('length joint set\n', jointset, "\n")
        for point in jointset:
            print('lenght point\n', point, "\n")
            joint_angle = calculate_angle((point[0], point[1]), (point[2], point[3]))
            joint_length = distance((point[0], point[1]), (point[2], point[3]))
            joint_length_list.append(joint_length)
            print('points : ', point)
            print('joint angle: ', joint_angle)
            print('joint length: ', joint_length)
            print()
        jointset_length_list.append(joint_length_list)
        
    print('jointset length list',jointset_length_list)    
    return jointset_length_list
    

    #joint set 간격 구하기
def joint_spacing(joint_points, lens) :
    print('get rho & spacing\n')
    jointset_spacings = []
    sorted_joint_points = []
    for i, jointset in enumerate(joint_points):
        # print("joint_spacing jointset: ", jointset)
        jointset_rho = []
        print("joint set",i)
 
        for point in jointset:
            # print('points\n', point)
            _,joint_rho = getpolar(point[0], point[1], point[2], point[3])
            jointset_rho.append(joint_rho)
            print('rho : ', joint_rho)
            
        sorted_arg = np.argsort(np.array(jointset_rho))
        print(sorted_arg)
        jointset_rho = np.sort(np.array(jointset_rho))
        sortedpoints = []
        for i in range (0,len(jointset)):
            sortedpoints.append(jointset[sorted_arg[i]])
        print('jointset',i,'rho:',jointset_rho)
        sorted_joint_points.append(sortedpoints)
        jointset_spacings.append(np.diff(jointset_rho))
        print()

    # print('jointset spacinig', jointset_spacings)
    real_length_list = []

    for i in range(len(jointset_spacings)):
        real_length = lens.real_length(jointset_spacings[i])
        # print('real joinset',i,'spacings', real_length)
        real_length_list.append(real_length)

    return real_length_list, sorted_joint_points


def make_data(joint_points, realDistance) :
    print('make Data\n', joint_points, "\n")
    data = []
    lens = Lens(realDistance * 10,12,6.287,4050)
    spacing,sorted_joint = joint_spacing(joint_points, lens)
    length = joint_length(joint_points, lens)
    
    
    print(length, spacing)
    
    # print(length[0][0])
    for num  in range(len(joint_points)) :
        # print("spacing: ", spacing)
        # print("sorted)joint: ", sorted_joint)
        # print("sorterd_joint: " , sorted_joint , "\n")
        
        # print('spacing\n', spacing, '\nlength\n', length, "\n")
        jointSetData = {
            "lines": [],
            "angles": [],
            "spacing": [],
            "length":  []
        }
        i = 0
        for point in sorted_joint[num]:
            jointSetData['lines'].append([[int(point[0]), int(point[1])], [int(point[2]),int(point[3])]])
            # jointSetData['length'].append(length[i])
            jointSetData['angles'].append(calculate_angle([point[0], point[1]], [point[2],point[3]]))
            i += 1
        jointSetData['spacing']=spacing[num]
        jointSetData['length']=length[num]
        

        data.append(jointSetData)

    return data


