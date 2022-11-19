from contextlib import nullcontext
import cv2
import os
import pandas as pd
import numpy as np
from Configuracao import Configuracao

class ProdutoDataSet:
    #global produtosJogosDf
    produtosJogosDf = None
    imageFeatures = 5000
    Config = None

    def __init__(self, config):
        listaNomeJogos = [
            "01_elden_ring",
            "02_uncharted_lost_legacy",
            "03_god_of_war",
            "04_resident_evil_origins",
            "05_demons_souls",
            "06_street_fighter_5",
            "07_heavy_rain",
            "08_killzone",
            "09_uncharted_4",
            "10_metal_gear_solid_5",
            "11_red_dead_redemption_2",
            "12_the_last_of_us_2",
            "13_gran_turismo_sport",
            "14_killer_instinct",
            "15_assassins_creed_black_flag",
            "16_call_of_duty_advanced_warfare",
            "17_the_last_of_us",
            "18_resident_evil_5",
            "19_gran_turismo_6",
            "20_halo_5",
            "21_halo_3",
            "22_halo_reach",
            "23_halo_4",
            "24_halo_anniversary_360",
            "25_fifa_13",
            "26_pes_2009",
            "27_fifa_15",
            "28_fifa_14"
        ]

        self.Config = config
        self.imageFeatures = config.NumeroCaracteristicas
        #global produtosJogosDf
        self.produtosJogosDf = pd.DataFrame([], columns=["codigo", "nome", "foto_frente", "foto_verso", "keypoints_frente_orb", "descriptor_frente_orb", "keypoints_verso_orb", "descriptor_verso_orb", "keypoints_frente_sift", "descriptor_frente_sift", "keypoints_verso_sift", "descriptor_verso_sift"])

        caminhoBase = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Imagens"))

        orb = cv2.ORB_create(nfeatures=self.imageFeatures)
        for nomeJogo in listaNomeJogos:
            partesNome = nomeJogo.split('_')
            indice = int(partesNome[0])
            nomeTratado = ""
            kpFrenteOrb = None
            kpVersoOrb = None
            descFrenteOrb = None
            descVersoOrb = None
            kpFrenteSift = None
            kpVersoSift = None
            descFrenteSift = None
            descVersoSift = None

            for indiceNome in range(1, len(partesNome)):
                nomeTratado += ("" if len(nomeTratado) == 0 else " ") + partesNome[indiceNome][0].upper() + partesNome[indiceNome][1:]

            if os.path.exists(os.path.join(caminhoBase, "Jogos\\" + nomeJogo + "_front.png")):
                imgFrente = cv2.imread(   os.path.join(caminhoBase, "Jogos\\" + nomeJogo + "_front.png") )
                if (self.Config.ResizeImagens):
                    imgFrente = cv2.resize(imgFrente, (400,400), interpolation = cv2.INTER_AREA)

                #imagemAux = cv2.cvtColor(imgVerso,cv2.COLOR_BGR2GRAY)
                kpFrenteOrb, descFrenteOrb = orb.detectAndCompute(imgFrente, None)

            if os.path.exists(os.path.join(caminhoBase, "Jogos\\" + nomeJogo + "_back.png")):
                imgVerso = cv2.imread(   os.path.join(caminhoBase, "Jogos\\" + nomeJogo + "_back.png") )
                if (self.Config.ResizeImagens):
                    imgVerso = cv2.resize(imgVerso, (400,400), interpolation = cv2.INTER_AREA)

                #imagemAux = cv2.cvtColor(imgVerso,cv2.COLOR_BGR2GRAY)
                kpVersoOrb, descVersoOrb = orb.detectAndCompute(imgVerso, None)

            self.produtosJogosDf.loc[self.produtosJogosDf.shape[0]] = [indice, nomeTratado, imgFrente, imgVerso, kpFrenteOrb, descFrenteOrb, kpVersoOrb, descVersoOrb, kpFrenteSift, descFrenteSift, kpVersoSift, descVersoSift]

    def ProcurarImagem(self, imagem):
        bruteforce = None
        flann = None
        kp1 = None
        desc1 = None

        orb = cv2.ORB_create(nfeatures=self.imageFeatures)
        kp1, desc1 = orb.detectAndCompute(imagem, None)

        if (desc1 is None):
            return [], np.zeros([300, 600,3],dtype=np.uint8), 0

        if (self.Config.Algoritimo == 0):
            # Inicialização Brute Force Matcher
            bruteforce = cv2.BFMatcher()


        elif (self.Config.Algoritimo == 1):
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH, 
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params,search_params)

        product = []
        closeMatches = []
        kpSeleconado = None
        productImage = np.zeros([100,100,3],dtype=np.uint8)
        
        for imgIndex in range(0, self.produtosJogosDf.shape[0]):
            productImageFront = self.produtosJogosDf.loc[imgIndex]["foto_frente"]
            productImageBack = self.produtosJogosDf.loc[imgIndex]["foto_verso"]

            descComparacaoFrente = self.produtosJogosDf.loc[imgIndex]["descriptor_frente_orb"]
            descComparacaoVerso = self.produtosJogosDf.loc[imgIndex]["descriptor_verso_orb"]
            kpFrente = self.produtosJogosDf.loc[imgIndex]["keypoints_frente_orb"]
            kpVerso = self.produtosJogosDf.loc[imgIndex]["keypoints_verso_orb"]
            
            matches = None
            if (not descComparacaoFrente is None):
                if (self.Config.Algoritimo == 0):
                    matches = self.CompararImagemBFM(bruteforce, descComparacaoFrente, desc1)
                elif (self.Config.Algoritimo == 1):
                    matches = self.CompararImagemFlann(flann, descComparacaoFrente, desc1)

                if (not matches is None and len(matches)> self.Config.NumeroMinimoCorrespondencias  and len(matches) > len(closeMatches)):
                    closeMatches = matches
                    kpSeleconado = kpFrente
                    product = self.produtosJogosDf.loc[imgIndex]
                    productImage = productImageFront
                
            matches = None
            if (not descComparacaoVerso is None):
                if (self.Config.Algoritimo == 0):
                    matches = self.CompararImagemBFM(bruteforce, descComparacaoVerso, desc1)
                elif (self.Config.Algoritimo == 1):
                    matches = self.CompararImagemFlann(flann, descComparacaoVerso, desc1)

                if (not matches is None and len(matches)> self.Config.NumeroMinimoCorrespondencias  and len(matches) > len(closeMatches)):
                    closeMatches = matches
                    kpSeleconado = kpVerso
                    product = self.produtosJogosDf.loc[imgIndex]
                    productImage = productImageBack

        #imagem = cv2.resize(imagem, (300,300), interpolation = cv2.INTER_AREA)
        imagemComparacao = np.zeros([300, 600,3],dtype=np.uint8)
        try:
            imagemComparacaoAux = cv2.drawMatchesKnn(imagem, kp1, productImage, kpSeleconado, closeMatches, None, flags=2)
        except:
            imagemComparacaoAux = productImage

        imagemComparacaoAux = cv2.resize(imagemComparacaoAux, (300,300), interpolation = cv2.INTER_AREA)
        imagemComparacao[0:imagemComparacaoAux.shape[0], 0:imagemComparacaoAux.shape[1]] = imagemComparacaoAux

        return product, imagemComparacao, len(closeMatches)

    def CompararImagemBFM(self, bruteforce, descComparacao, descBase):
        matches = bruteforce.knnMatch(descBase, descComparacao, k=2)
        
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        return good

    def CompararImagemFlann(self, flann, descComparacao, descBase):
        good = []
        try:
            matches = flann.knnMatch(descComparacao,descBase,k=2)
        
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append([m])
        except:
            good = []

        return good
