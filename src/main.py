import argparse
import os
import sys
import time
from io import BytesIO

import cv2
import requests
from matplotlib import pyplot as plt
from PIL import Image

from displayresult import (drawAll, drawJointset, getDataFrame, makeStereonet,
                           saveDataFrameAsImage)
from function import evaluate, imageProcessing
from jointExtract import getLine, make_data, redImage

parser = argparse.ArgumentParser()
parser.add_argument('-imgURL', help=' : Please set the s3 image URL')
parser.add_argument('-distance', help=' : Plsease set distance')


def main(args):
    print('start')
    start = time.time()

    ####################################################################################
    ################################## 이미지 경로 #######################################
    # img_url = "https://bclops.s3.ap-northeast-2.amazonaws.com/input.png"
    # img_url1 = "https://bclops.s3.ap-northeast-2.amazonaws.com/input_1.png"
    # img_url2 = "https://bclops.s3.ap-northeast-2.amazonaws.com/input_2.png"
    # img_url3 = "https://bclops.s3.ap-northeast-2.amazonaws.com/input_3.png"
    # img_url4 = "https://bclops.s3.ap-northeast-2.amazonaws.com/input_4.png"
    ####################################################################################
    ####################################################################################
    print(args)
    img_url = args.imgURL
    # print(argv[1])
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((1024, 512))
    img.save('./evalutate/image/input.png')

    original_image = "./evalutate/image/input.png"
    ai_image = evaluate()
    print("ai fin")
    output = imageProcessing(original_image, ai_image)
    cv2.imwrite("output.jpg", output)
    # cv2.imshow("output", output)

    # 카메라팀
    redImageResult = redImage(output)
    jointPoint = getLine(redImageResult)
    distance = int(args.distance)
    data = make_data(jointPoint, distance)

    original_img = cv2.imread(original_image)
    resultImg = drawAll(original_img, data)

    cv2.imwrite("resultimg.jpg", resultImg)
    joint_setlist = []
    for i in range(0, len(data)):
        jointset_result = drawJointset(original_img, data, "jointset%d" % i, setnum=i)
        joint_setlist.append(jointset_result)
        # cv2.imshow("resultimg%d" % i, jointset_result)
        cv2.imwrite("resultjointset%d.jpg" % i, jointset_result)
    plt.close('all')
    stereonet = makeStereonet(data)
    plt.savefig('stereonetImg.jpg')
    # plt.show()
    dataFrame = getDataFrame(data)
    print(dataFrame)
    dataFrame.to_csv("resultData.csv",encoding = 'cp949')
    saveDataFrameAsImage(dataFrame, "table.jpg")

    stop = time.time()
    print("testing time :", round(stop - start, 3), "ms")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("testing time :", round(stop - start, 3), "ms")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    #return 값 필요하시면 resultData.csv, resultimg.jpg, stereonetImg.jpg, output.jpg, resultJoint0~(len(joint_setlist)-1)까지 이미지 여시고 적당히 인코딩해서 사용하시면 됩니다.
    
if __name__ == '__main__' :
    #cli환경에서 받는 방식입니다 python main.py -imgURL "s3url파라미터"(string) -distance 거리파라미터(int)
    #만약 함수형으로 받으시는거면 if __name__~끝까지 다 주석처리 하시고 (imgURL="s3url파라미터"(string), distance = 거리파라미터(int))형태로 main 함수 실행해주세요
    argv = sys.argv
    args = parser.parse_args()
    main(args)
