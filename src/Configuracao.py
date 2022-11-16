from tkinter import *
from tkinter import ttk
import cv2


class Configuracao:

    janela = None
    CmbWebcam = None
    txtIPCamera = None
    CmbAlgoritimo = None
    LblMessage = None
    txtNumeroCorrespondecias = None
    CmbCompactarImagens = None

    Iniciar = False
    WebCamIndex = -1
    IPCamera = ""
    Algoritimo = -1
    NumeroMinimoCorrespondencias = 50
    ResizeImagens = False

    def __init__(self, ):
        rowIndex = 0
        self.janela = Tk()
        self.janela.title("Detecção de produtos")
        self.janela.resizable(width=False, height=False)

        frame = Frame(self.janela, width=500, height=260)
        frame.grid(row=0, column=0, padx=10, pady=5)
        
        label = ttk.Label(frame, text="WebCam:").grid(row=rowIndex, column=0, padx=5, pady=5, sticky="E")
        #label.pack()

        webcamIndex = 0
        arrWebCams = []
        while (True):
            webcam = cv2.VideoCapture(webcamIndex)

            if not webcam.read()[0]:
                break
            else:
                arrWebCams.append("Webcam " + str(webcamIndex))

            webcam.release()

            webcamIndex += 1
        arrWebCams.append("IP / arquivo de vídeo")

        self.CmbWebcam = ttk.Combobox(frame, width=35)
        self.CmbWebcam.grid(row=rowIndex, column=1, padx=5, pady=5, sticky="W")
        self.CmbWebcam['values'] = arrWebCams
        self.CmbWebcam['state'] = 'readonly'
        self.CmbWebcam.current(0)

        rowIndex += 1
        label = ttk.Label(frame, text="IP Câmera/ Caminho do arquivo:")
        label.grid(row=rowIndex, column=0, padx=5, pady=5, sticky="E")
        self.txtIPCamera = Entry(frame, width=35)
        self.txtIPCamera.grid(row=rowIndex, column=1, padx=5, pady=5, sticky="W")
        self.txtIPCamera.insert(0, "http://192.168.15.16:4747/video")

        rowIndex += 1
        label = ttk.Label(frame, text="Algoritmo:").grid(row=rowIndex, column=0, padx=5, pady=5, sticky="E")
        self.CmbAlgoritimo = ttk.Combobox(frame, width=35)
        self.CmbAlgoritimo.grid(row=rowIndex, column=1, padx=5, pady=5, sticky="W")
        self.CmbAlgoritimo['values'] = ["Brute Force Matcher", "FLANN", "Nenhum"]
        self.CmbAlgoritimo['state'] = 'readonly'
        self.CmbAlgoritimo.current(0)

        rowIndex += 1
        label = ttk.Label(frame, text="Número Correspondências:").grid(row=rowIndex, column=0, padx=5, pady=5, sticky="E")
        self.txtNumeroCorrespondecias =  Entry(frame)
        self.txtNumeroCorrespondecias.grid(row=rowIndex, column=1, padx=5, pady=5, sticky="W")
        self.txtNumeroCorrespondecias.insert(0, "50")

        rowIndex += 1
        label = ttk.Label(frame, text="Compactar Imagens:").grid(row=rowIndex, column=0, padx=5, pady=5, sticky="E")
        self.CmbCompactarImagens = ttk.Combobox(frame, width=35)
        self.CmbCompactarImagens.grid(row=rowIndex, column=1, padx=5, pady=5, sticky="W")
        self.CmbCompactarImagens['values'] = ["Sim", "Não"]
        self.CmbCompactarImagens['state'] = 'readonly'
        self.CmbCompactarImagens.current(1)

        self.LblMessage = ttk.Label(self.janela, text="", foreground="red")
        self.LblMessage.grid(row=1, column=0, padx=5, pady=5)

        label = ttk.Label(self.janela, text="")
        label.grid(row=2, column=0, padx=5, pady=5)

        btnIniciar = ttk.Button(self.janela, text = "Iniciar", command=lambda: Configuracao.Iniciar_Click(self))
        btnIniciar.grid(row=3, column=0, padx=5, pady=5)

        self.janela.geometry("450x300")
        self.janela.eval('tk::PlaceWindow . center')
        self.janela.mainloop()

    def Iniciar_Click(self, ):
        if (self.CmbWebcam.current() < 0):
            self.LblMessage['text'] = "Nenhuma webcam selecionada"
            return

        self.WebCamIndex = self.CmbWebcam.current()

        if ( self.CmbWebcam.current() == len(self.CmbWebcam["values"])-1 ):
            self.WebCamIndex = -1
            if (len(self.txtIPCamera.get()) == 0):
                self.LblMessage['text'] = "IP da camera não informado"
                return
            self.IPCamera = self.txtIPCamera.get()

        if (self.CmbAlgoritimo.current() < 0):
            self.LblMessage['text'] = "Nenhum algorítmo selecionado"
            return

        self.Algoritimo = self.CmbAlgoritimo.current()

        if (len(self.txtNumeroCorrespondecias .get()) == 0):
            self.LblMessage['text'] = "Número de correspondências não preenchido"
            return
        self.NumeroMinimoCorrespondencias = int(self.txtNumeroCorrespondecias .get())

        if (self.CmbCompactarImagens.current() == 0):
            ResizeImagens = True
        else:
            ResizeImagens = False

        self.Iniciar = True

        self.janela.destroy()
