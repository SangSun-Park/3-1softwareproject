from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
import os
import time

import re
import face_detect as fd

notAllowedChar = '^[/:*?"<>|\\\\ ]+$'


def Allow_Certain_Folder_Name(string):
    # 폴더에 넣을 수 없는 문자 및 .으로 이루어진 값들은 넘어감
    st = re.match(notAllowedChar, string)
    # 안에 잘못된값이 없을 경우
    if not bool(st):
        # 점 또는 스페이스바로만 이루어진 값도 패스
        rpStr = string.replace(" ", "")
        rpStr = rpStr.replace(".", "")
        if rpStr is "":
            return False
    else:
        return False

    return True


def Crawling_Image(search, maxAmount):
    if not Allow_Certain_Folder_Name(search):
        return

    for c in '/:*?"<>|\\ ':
        search = search.replace(c, "")

    sstart = time.time()
    names = search.split(',')
    # 이미지 url https://search.naver.com/search.naver?where=image&query=검색이름
    # 네이버는 기본적으로 50개를 불러오는 방식을 이용함
    # 이 때 start 인자를 이용하면 시작지점을 정할 수 있어 50개단위로 여러번 작동시켜 원하는만큼 받아오도록 함

    # 원래 구글을 이용하려 하였으나 구글을 통해 받은 이미지 해상도가 너무 좋지 않아 네이버로 변경
    # 또한 네이버는 이미지 소스에 data-source가 있는 경우를 통해 제한할 경우 구글처럼 로고가 포함되는 경우가 없어 오류 조정에도 더 편함
    # 한국 인물 특성상 네이버를 통해서도 많은 이미지를 얻을 수 있기 때문에 네이버로 이용
    for name in names:
        naverUrl = "https://search.naver.com/search.naver?"

        startNum = 0
        currentImageAmount = 1

        dirName = "data/IMG/" + name + "/"
        os.path.exists(dirName)
        try:
            if os.path.exists(dirName):
                start = time.time()
                files = [f for f in os.listdir(dirName) if os.path.isfile(os.path.join(dirName, f))]
                for i, file in enumerate(files):
                    os.remove(dirName + "IMG" + f"{i + 1:05}" + ".jpg")
                end = time.time()
                print(f"Remove All Files :{end - start: .2f}s")
            else:
                os.makedirs(dirName)
                print("Create Directory: " + dirName)

        except OSError:
            print("Error: Creating directory: " + dirName)

        # 기본적으로 요청량만큼 받지만 이미지 다운로드에 실패하여 startNum이 너무 늘어날 경우 그냥 끝냄
        while currentImageAmount <= maxAmount and startNum < maxAmount * 2:
            params = {
                "query": name,
                "where": "image",
                "start": startNum
            }

            htmlData = requests.get(naverUrl, params)
            if htmlData.status_code == 200:
                soup = BeautifulSoup(htmlData.text, 'html.parser')
                imgs = soup.find_all('img', {'data-source': True})
                # 검색을 해서 만약에 검색결과가 안나오는 경우 실행을 안함
                if imgs is not None:
                    for i in enumerate(imgs):
                        try:
                            img = urlopen(i[1].attrs['data-source']).read()
                            filename = dirName + 'IMG' + f'{currentImageAmount:05}' + '.jpg'
                            tempFileName = dirName + 'Temp' + f'{currentImageAmount:05}' + '.jpg'
                            with open(tempFileName, "wb") as f:
                                f.write(img)
                                f.close()

                            if fd.find_one_face_dnn(tempFileName, filename):
                                print(i[1].attrs['alt'])
                                print("Img Save Success: " + str(currentImageAmount))
                                currentImageAmount += 1

                            if currentImageAmount > maxAmount:
                                break
                        except ValueError:
                            continue

            startNum += 50

    end = time.time()
    print(f"Crawling Complete :{end - sstart: .2f}s")

    # 이미지 url https://www.google.com/search?q=검색내용&tbm=isch
    # 구글은 기본적으로 20개를 불러오는 방식을 이용함
    # 이 때 start 인자를 이용하면 시작지점을 정할 수 있어 20개단위로 여러번 작동시켜 원하는만큼 받아오도록 함
    '''
    googleUrl = "https://www.google.com/search?"

    startNum = 0
    currentImageAmount = 1

    # 기본적으로 요청량만큼 받지만 이미지 다운로드에 실패하여 startNum이 너무 늘어날 경우 그냥 끝냄
    while currentImageAmount <= maxAmount and startNum < maxAmount * 3:
        params = {
            "q": name,
            "tbm": "isch",
            "start": startNum
        }

        htmlData = requests.get(googleUrl, params)

        # 데이터 불러오는데에 성공했을 경우에 사용
        if htmlData.status_code == 200:
            soup = BeautifulSoup(htmlData.text, "html.parser")
            imgDatas = soup.find_all("img")
            for img in imgDatas:
                print (img)

            dirName = "data/IMG/" + name + '/'

            try:
                if not os.path.exists(dirName):
                    os.makedirs(dirName)
                    print("Create Directory: " + dirName)
            except OSError:
                print("Error: Creating directory: " + dirName)

            for i in enumerate(imgDatas):
                # 구글 이미지 소스 파일 중에는 구글로고가 포함되어있는데 이는 urlopen으로 열리지 않는 이미지이다
                # 따라서 ValueError로 예외처리 해줘서 문제가 안생기도록 한다
                try:
                    img = urlopen(i[1].attrs['src']).read()
                    filename = dirName + name + str(currentImageAmount) + '.jpg'
                    with open(filename, 'wb') as f:
                        f.write(img)
                        print(i[1])
                        print("Img Save Success: " + str(currentImageAmount))
                        currentImageAmount += 1
                except ValueError:
                    continue

        startNum += 20
    '''
