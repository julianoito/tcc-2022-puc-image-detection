import sys
import cv2
from Produtos import ProdutoDataSet
import numpy as np
import os
from Configuracao import Configuracao


print ("Iniciando configurações")
configWindows = Configuracao()

DataSetProdutos = ProdutoDataSet(configWindows)
DataSetProdutos.Config = configWindows

if (not configWindows.Iniciar):
    print ("Execução interrompida")
    sys.exit()


if (configWindows.WebCamIndex == -1):
    webcam = cv2.VideoCapture(configWindows.IPCamera)
else:
    webcam = cv2.VideoCapture(configWindows.WebCamIndex, cv2.CAP_DSHOW)

camWidth = 1280
camHeight = 720
screenSize = [640, 480]
frameSkipCheck = 15
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)


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