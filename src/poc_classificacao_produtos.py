import sys
import cv2
from Produtos import ProdutoDataSet
import numpy as np
import os
from Configuracao import Configuracao
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime



BoundBoxLock = threading.Lock()

class VideoCaptureThread(threading.Thread):
    ResultWindow = None
    def __init__(self, resultWindow):
        threading.Thread.__init__(self)
        self.ResultWindow = resultWindow
    def run(self):
        if (configWindows.WebCamIndex == -1):
            webcam = cv2.VideoCapture(configWindows.IPCamera)
        else:
            webcam = cv2.VideoCapture(configWindows.WebCamIndex, cv2.CAP_DSHOW)

        camWidth = 1920
        camHeight = 1080
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)

        imageAnalysisThread = ImageAnalysisThread(self.ResultWindow)
        imageAnalysisThread.daemon = True
        imageAnalysisThread.start()

        if (webcam.isOpened()):
            frameCapturado, frame = webcam.read()

            while (appRunning and frameCapturado):
                key = cv2.waitKey(10)
                frameCapturado, frame = webcam.read()
                
                cv2.namedWindow('videoSoruce',cv2.WND_PROP_AUTOSIZE)
                cv2.setWindowProperty('videoSoruce', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.moveWindow("videoSoruce",0,0)

                if (imageAnalysisThread.ImageToProcess is None):
                    imageAnalysisThread.ImageToProcess = frame

                boundBoxesAux = []
                with BoundBoxLock:
                    if (imageAnalysisThread.CurrentBoundBoxProdutcs is not None and len(imageAnalysisThread.CurrentBoundBoxProdutcs)>0):
                        boundBoxesAux = imageAnalysisThread.CurrentBoundBoxProdutcs

                if (len(boundBoxesAux)>0):
                    for i in range(0, len(boundBoxesAux)):
                        product = boundBoxesAux[i]

                        if (product[5] is not None):
                            bbColor = (0,255,0)
                            textColor = (255,255,255)
                            productName = product[5]["nome"]
                            if (product[1]-30 > 0 and product[1]-1 > 0):
                                cv2.rectangle(frame, (product[0],product[1]-30), (product[2], product[1]-1), color=bbColor, thickness=-1)
                                cv2.putText(frame, productName, (product[0]+5,product[1]-10), cv2.FONT_HERSHEY_PLAIN, 1, textColor, 2)
                        else:
                            bbColor = (200,200,200)
                       
                        cv2.rectangle(frame, (product[0],product[1]), (product[2], product[3]), color=bbColor, thickness=1)

                cv2.imshow("videoSoruce", frame)
                
        webcam.release()
        cv2.destroyAllWindows()
        imageAnalysisThread.join()


class ImageAnalysisThread(threading.Thread):
    lastDatetimeCheck = None
    lblLastCheckDuration = None
    lblLastCheck = None
    resultList = []
    ImageToProcess = None
    ResultWindow = None
    LastProductImg = None

    #CurrentBoundBox = []
    CurrentBoundBoxProdutcs = []

    def __init__(self, resultWindow):
        threading.Thread.__init__(self)
        self.ResultWindow = resultWindow

    def run(self):
        self.lblLastCheckDuration = ttk.Label(self.ResultWindow, text="Check time: ", font=('Arial', 9))
        self.lblLastCheckDuration.pack(side="top", anchor="w")
        self.lblLastCheck = ttk.Label(self.ResultWindow, text="Last Check: ", font=('Arial', 9))
        self.lblLastCheck.pack(side="top", anchor="w")
        self.lastDatetimeCheck = datetime.now()

        detectionModel = "../model_data/frozen_inference_graph.pb"
        detectionModelConfigPath = "../model_data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        netDetectionModel = cv2.dnn_DetectionModel(detectionModel, detectionModelConfigPath)
        netDetectionModel.setInputSize(320, 320)
        netDetectionModel.setInputScale(1.0/127.5)
        netDetectionModel.setInputMean((127.5, 127.5, 127.5))
        netDetectionModel.setInputSwapRB(True)

        with open('../model_data/coco.names', 'r') as f:
            detectionModelClassesList = f.read().splitlines()
            f.close()

        detectionModelClassesList.insert(0, "__Background__")

        while (appRunning):
            if (not self.ImageToProcess is None):
                classLabelIDs, confidences, bboxes = netDetectionModel.detect(self.ImageToProcess, confThreshold=0.4)
                bboxes = list(bboxes)
                confidences = list(np.array(confidences).reshape(1,-1)[0])
                confidences = list(map(float, confidences))
                
                bboxIdx = cv2.dnn.NMSBoxes(bboxes, confidences, score_threshold = 0.5, nms_threshold = 0.2)

                NewBoundBoxProdutcs = []
                if (len(bboxIdx)!=0):
                    for i in range(0, len(bboxIdx)):
                        bbox = bboxes[np.squeeze(bboxIdx[i])]
                        x,y,w,h = bbox

                        productImage = self.ImageToProcess[y:y+h, x:x+w]
                        productImage = cv2.cvtColor(productImage, cv2.COLOR_BGR2GRAY)

                        NewBoundBoxProdutcs.append([x, y, x+w, y+h, productImage, None])



                #if (len(self.CurrentBoundBox) == 0):
                    #self.CurrentBoundBox = NewBoundBoxProdutcs
                currentDate = datetime.now()
                dateDiff = currentDate - self.lastDatetimeCheck
                self.lastDatetimeCheck = currentDate
                self.lblLastCheck['text'] = "Last Check: " + currentDate.strftime("%d/%m/%Y %H:%M:%S")
                self.lblLastCheckDuration['text'] = "Check time: " + str(round(dateDiff.total_seconds(),4))  + "(s)"

                for label in self.resultList:
                    try:
                        label.destroy()
                    except:
                        continue
                for i in range (0, len(NewBoundBoxProdutcs)):
                    bbox = NewBoundBoxProdutcs[i]

                    product, imgProdutoComparacao, matchesCount, maxDescriptions = DataSetProdutos.ProcurarImagem(bbox[4])

                    if (appRunning and len(product) > 0):
                        labelProduto = ttk.Label(self.ResultWindow, text=product["nome"] + " (" + str(matchesCount) + "/" + str(maxDescriptions) + ")", font=('Arial', 10), background="blue", foreground="white")
                        labelProduto.pack(side="top", fill=X)
                        self.resultList.append(labelProduto)

                        #imgProdutoComparacao = cv2.resize(imgProdutoComparacao, (200,200), interpolation = cv2.INTER_AREA)
                        self.LastProductImg = imgProdutoComparacao
                        imgProduct = []
                        try:
                            blue,green,red = cv2.split(product["foto_frente"])
                            imgProdutoComparacao = cv2.merge((red,green,blue))
                            imgProdutoComparacao = cv2.resize(imgProdutoComparacao, (200,200), interpolation = cv2.INTER_AREA)

                            imgProduct.append(Image.fromarray(imgProdutoComparacao))
                            imgProduct[len(imgProduct)-1] = ImageTk.PhotoImage(image=imgProduct[len(imgProduct)-1])

                            labelProduto = ttk.Label(image=imgProduct[len(imgProduct)-1])
                            labelProduto.pack(side="top")
                            self.resultList.append(labelProduto)
                        except Exception as e:
                            print(str(e))

                        labelProduto = ttk.Label(self.ResultWindow, text="---------------", font=('Arial', 9))
                        labelProduto.pack(side="top")
                        self.resultList.append(labelProduto)
                        bbox[5] = product

                with BoundBoxLock:
                    self.CurrentBoundBoxProdutcs = NewBoundBoxProdutcs
                self.ImageToProcess = None
                time.sleep(1.01)
            else:
                time.sleep(0.01)


configWindows = Configuracao()

if (not configWindows.Iniciar):
    sys.exit()




## Teste nova janela
appRunning = True
mainWindow = Tk()
mainWindow.title("Busca de produtos")
mainWindow.geometry("1480x720+0+0")
labelLoading = ttk.Label(mainWindow, text="Carregando cache de produtos", font=('Arial', 16))
labelLoading.pack(side="top")
mainWindow.update()

DataSetProdutos = ProdutoDataSet(configWindows)
DataSetProdutos.Config = configWindows

labelLoading.destroy()
mainWindow.geometry("200x720+2000+0")

videoCaptureThread = VideoCaptureThread(mainWindow)
videoCaptureThread.daemon = True
videoCaptureThread.start()

mainWindow.mainloop()
appRunning = False

print("Esperando termino da thread videoCaptureThread")
#videoCaptureThread.join()

sys.exit()

##Fim Teste nova janela







count = frameSkipCheck
lastCheck = None
imgProductComparison = None
frameShape = None
frameComparacao = None





if (webcam.isOpened()):
    frameCapturado, frame = webcam.read()

    barraTitulo = np.zeros([20, 320,3],dtype=np.uint8)

    screen = np.zeros([screenSize[1], screenSize[0]+300,3],dtype=np.uint8)
    #print(screen.shape)

    while (frameCapturado):
        frameCapturado, frame = webcam.read()

        if (frameCapturado):

            if (count >= frameSkipCheck):
                product, imgProdutoComparacao, matchesCount = DataSetProdutos.ProcurarImagem(frame)
                lastCheck = imgProdutoComparacao
                count = 0

                imgProdutoComparacao = cv2.resize(imgProdutoComparacao, (300,300), interpolation = cv2.INTER_AREA)
                
                if (len(product) > 0):
                    imgProdutoComparacao[0:30, 0:300] = 0
                    cv2.putText(img = imgProdutoComparacao, text=product["nome"] + "(" + str(matchesCount) + ")", org=(5,15), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.85, color=(255, 255, 255),thickness=1)
                screen[0:300, screenSize[0]:screenSize[0]+300] = imgProdutoComparacao

            frame = cv2.resize(frame, (screenSize[0],screenSize[1]), interpolation = cv2.INTER_AREA)
            screen[0:frame.shape[0], 0:frame.shape[1]] = frame

            screen[0:20, 0:320] = 0
            cv2.putText(img = screen, text="ESC-Sair", org=(5,15), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.85, color=(255, 255, 255),thickness=1)
            cv2.imshow("Teste", screen)

        key = cv2.waitKey(5)
        count += 1
        

        if (key==27):
            break
        elif (key==32):
            cv2.imwrite(os.path.join(os.path.dirname(__file__), "Temp\\imagem.png"), lastCheck)
        elif (key > 0):
            print ("Tecla:" + str(key))

webcam.release()
cv2.destroyAllWindows()


