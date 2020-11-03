import os
import pickle

import tkinter.font as tkf
from tkinter import *
from tkinter import filedialog

import MovieEditor as me
import WebCrawler as wc
from face_learn import face_learning


class MainFrame(Frame):
    # filePath와 savePath가 변수설정

    def __init__(self, master):
        Frame.__init__(self, master)

        self.master = master
        self.master.title("FACESCAPE")
        self.pack(fill=BOTH, expand=True)
        self.names = []

        # 빈공간
        emptyFrame = Frame(self)
        emptyFrame.pack(fill=X)
        nameFont = tkf.Font(size=15, weight="bold")
        emptyLabel = Label(emptyFrame, text="FACESCAPE", font=nameFont)
        emptyLabel.pack(pady=10)

        # 크롤링
        crawlFrame = Frame(self)
        crawlFrame.pack(fill=NONE)
        crawlDirLbl = Label(crawlFrame, text="검색 이름: ", width=10, height=1)
        self.crawlName = StringVar()
        crawlDirInput = Entry(crawlFrame, textvariable=self.crawlName)
        crawlDirButton = Button(crawlFrame, text="Search", width=7, height=1, command=self.Search_Name, repeatdelay=100)
        crawlDirLbl.pack(side=LEFT, padx=8, pady=10)
        crawlDirInput.pack(side=LEFT, padx=3, pady=10)
        crawlDirButton.pack(side=LEFT, padx=4, pady=10)

        # 학습
        learnFrame = Frame(self)
        learnFrame.pack(fill=NONE)
        learnButton = Button(learnFrame, text="학습", width=8, height=1, command=self.Learning_Picture, repeatdelay=100)
        #delDirButton = Button(learnFrame, text="이미지삭제", width=8, height=1, command=self.Remove_IMGs, repeatdelay=100)
        #delDirButton.pack(side=LEFT, padx=8, pady=5)
        learnButton.pack(side=LEFT, padx=8, pady=5)

        # 인물 이름
        findFrame = Frame(self)
        findFrame.pack(fill=NONE)
        findLbl = Label(findFrame, text="편집 대상: ", width=10, height=1)
        self.findName = StringVar()
        findInput = Entry(findFrame, textvariable=self.findName)
        emptyLbl = Label(findFrame, width=7, height=1)
        findLbl.pack(side=LEFT, padx=8, pady=10)
        findInput.pack(side=LEFT, padx=3, pady=10)
        emptyLbl.pack(side=LEFT, padx=4, pady=10)

        # 파일 위치
        fileFrame = Frame(self)
        fileFrame.pack(fill=NONE)
        fileDirLbl = Label(fileFrame, text="파일 위치: ", width=10, height=1)
        self.filePath = StringVar()
        fileDirInput = Entry(fileFrame, textvariable=self.filePath)
        fileDirButton = Button(fileFrame, text="Find", width=7, height=1, command=self.Load_Movie_File, repeatdelay=100)
        fileDirLbl.pack(side=LEFT, padx=8, pady=10)
        fileDirInput.pack(side=LEFT, padx=3, pady=10)
        fileDirButton.pack(side=LEFT, padx=4, pady=10)

        # 저장 위치
        saveFrame = Frame(self)
        saveFrame.pack(fill=NONE)
        saveDirLbl = Label(saveFrame, text="저장 위치: ", width=10, height=1)
        self.savePath = StringVar()
        saveDirInput = Entry(saveFrame, textvariable=self.savePath)
        saveDirButton = Button(saveFrame, text="Find", width=7, height=1, command=self.Set_Output_Directory,
                               repeatdelay=100)
        saveDirLbl.pack(side=LEFT, padx=8, pady=10)
        saveDirInput.pack(side=LEFT, padx=3, pady=10)
        saveDirButton.pack(side=LEFT, padx=4, pady=10)

        # 파일 이름
        saveNameFrame = Frame(self)
        saveNameFrame.pack(fill=NONE)
        saveNameLbl = Label(saveNameFrame, text="파일 이름: ", width=10, height=1)
        self.saveFileName = StringVar()
        self.saveFileName.set("Output")
        saveNameInput = Entry(saveNameFrame, textvariable=self.saveFileName, width=20)
        saveNameLbl.pack(side=LEFT, padx=8, pady=10)
        saveNameInput.pack(side=LEFT, padx=3, pady=10)

        # 실행 버튼
        activateFrame = Frame(self)
        activateFrame.pack(fill=NONE)
        self.isActivated = False
        activateButton = Button(activateFrame, text="실행", width=7, height=1, command=self.Activate, repeatdelay=100)
        activateButton.pack(padx=4, pady=3)

        # 실행 상태
        progressFrame = Frame(self)
        progressFrame.pack(fill=BOTH)
        self.progressMessage = StringVar()
        progressLabel = Label(progressFrame, textvariable=self.progressMessage, relief=SOLID, bd=1,
                              bg='ghost white', width=50, height=30)
        progressLabel.pack(side=TOP, anchor=N, padx=5, pady=5)

    # 검색해서 이미지 저장하기
    def Search_Name(self):
        if not self.crawlName.get() == "" or None:
            if wc.Allow_Certain_Folder_Name(self.crawlName.get()):
                wc.Crawling_Image(self.crawlName.get(), 100)
            else:
                self.Set_Progress_Message('검색어에 \%/:*?"<>|.를 넣을 수 없습니다.')
        else:
            self.Set_Progress_Message("검색어를 입력해주세요")

    # 이미지 내용물 삭제
    def Remove_IMGs(self):
        a = 1

    # 이미지 학습
    def Learning_Picture(self):
        face_learning()

    # 변환할 동영상 파일
    def Load_Movie_File(self):
        path = filedialog.askopenfilename(
            initialdir="C:",
            filetypes=([('Video Files(mkv, avi, mp4, mpg, flv, wmv)', '*.mkv;*.avi;*.mp4;*.mpg;*.flv;*.wmv')]))
        self.filePath.set(path)

    # 저장할 폴더
    def Set_Output_Directory(self):
        path = filedialog.askdirectory(
            initialdir="C:")
        self.savePath.set(path)

    # 실행시키기
    def Activate(self):
        checkFileMsg = me.Check_File(self.filePath.get())
        checkDirMsg = me.Check_Directory(self.savePath.get())

        if checkFileMsg is not None:
            self.Set_Progress_Message(checkFileMsg)

        elif checkDirMsg is not None:
            self.Set_Progress_Message(checkDirMsg)

        elif self.saveFileName.get() == " " or None:
            self.Set_Progress_Message("파일 이름이 설정되지 않았습니다.")

        else:
            if self.Check_People() is False:
                self.Set_Progress_Message("학습된 인물인지 확인해주세요.")
                return

            self.progressMessage.set("실행 중")
            saveFilePath = self.savePath.get() + "/" + self.saveFileName.get()
            extension = me.Find_Extension(self.filePath.get())
            me.Edit_Movie(self.filePath.get(), saveFilePath, extension, self.names)
            self.progressMessage.set("실행 완료")
            os.startfile(self.savePath.get())

    # 인물이 학습이 됐는지 확인
    def Check_People(self):
        learned_path = 'data/modellist.bin'
        if not os.path.exists(learned_path):
            return False

        names = self.findName.get().replace(' ', '')
        if self.findName.get() == "" or None:
            return False

        names = names.split(',')

        f = open('data/modellist.bin', "rb")
        learned = pickle.load(f)
        for name in names:
            if name not in learned:
                return False

        self.names = names
        f.close()
        return True

    def Set_Progress_Message(self, msg):
        self.progressMessage.set(msg)


def MainGUI():
    root = Tk()

    screenWidth = root.winfo_screenwidth()
    screenHeight = root.winfo_screenheight()
    width = 480
    height = 400
    pos_x = int(screenWidth / 2 - width / 2)
    pos_y = int(screenHeight / 2 - height / 2)
    root.geometry(str(width) + "x" + str(height) + "+" + str(pos_x) + "+" + str(pos_y))

    MainFrame(root)

    root.resizable(False, False)
    root.mainloop()


def main():
    MainGUI()


if __name__ ==  "__main__":
    main()
