import cv2
import os
import pandas as pd

class ProdutoDataSet:
    produtosJogosDf = None

    def __init__(self, ):
        listaNomeJogos = [
            "01_elden_ring",
            "02_uncharted_lost_legacy",
            "03_god_of_war.png",
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
        global produtosJogosDf
        produtosJogosDf = pd.DataFrame([], columns=["codigo", "nome", "foto_frente", "foto_verso"])

        caminhoBase = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Imagens"))

        for nomeJogo in listaNomeJogos:
            partesNome = nomeJogo.split('_')
            indice = int(partesNome[0])
            nomeTratado = ""
            for indiceNome in range(1, len(partesNome)):
                nomeTratado += ("" if len(nomeTratado) == 0 else " ") + partesNome[indiceNome][0].upper() + partesNome[indiceNome][1:]

            if os.path.exists(os.path.join(caminhoBase, "Jogos\\" + nomeJogo + "_front.png")):
                imgFrente = cv2.imread(   os.path.join(caminhoBase, "Jogos\\" + nomeJogo + "_front.png") )

            if os.path.exists(os.path.join(caminhoBase, "Jogos\\" + nomeJogo + "_back.png")):
                imgVerso = cv2.imread(   os.path.join(caminhoBase, "Jogos\\" + nomeJogo + "_back.png") )

            produtosJogosDf.loc[produtosJogosDf.shape[0]] = [indice, nomeTratado, imgFrente, imgVerso]


DataSetProdutos = ProdutoDataSet()