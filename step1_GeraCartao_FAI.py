# -*- coding: utf-8 -*-
"""
Automatização de Simulações de FAIs no ATP pelo Python (versão OOP)
Autora: Gabriela Nunes  |  e-mail: nuneslopesgabriela@gmail.com
Data original: 12/12/2022
Update: 09/10/2025

Descrição:
    - loop_completo() executa todos os passos na mesma ordem.
    - Pode ser executado diretamente: python step1_GeraCartao_FAI.py
"""

from __future__ import annotations

# Imports das funções originais ------------------------------
# Fallback para facilitar execução local sem mexer em sys.path:
try:
    from Functions.lib_automatiza34barras import (
        mudacarregamento,
        salvacartaonapasta,
        faino34barras_HIFreal,
        faino34barras_HIFmodelo2022,
        muda_potencia_GD,
        desconecta_chaves,
        parametrizamodelo
    )
except ModuleNotFoundError:
    # Se o pacote "Functions" não estiver no PYTHONPATH, tenta importar o módulo local.
    from Functions.lib_automatiza34barras import (
        mudacarregamento,
        salvacartaonapasta,
        faino34barras_HIFreal,
        faino34barras_HIFmodelo2022,
        muda_potencia_GD,
        desconecta_chaves,
        parametrizamodelo
    )

# Bibliotecas padrão ----------------------------------------
import time
from typing import List
import numpy as np
import pandas as pd
from os import listdir  # (mantido pois está no original, mesmo não usado no trecho)
from pathlib import Path

class Automatiza34Barras:
    """
    Classe que encapsula a automação, mantendo a lógica original.
    """

    def __init__(self) -> None:
        # ---------------------------
        # Configurações iniciais
        # ---------------------------
        # (mantidas as mesmas listas e valores do código fornecido)
        self.Fases: List[str] = ['FaseA', 'FaseB', 'FaseC']

        self.barrasfasea: List[int] = [
            2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34
        ]
        self.barrasfaseb: List[int] = [
            2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18,
            19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
        ]
        self.barrasfasec: List[int] = [
            2, 3, 4, 5, 6, 7, 8, 11, 14, 16, 17, 19,
            20, 21, 22, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34
        ]

        self.cargasp: List[str] = [
            'nenhum', 'CP01', 'CP02', 'CP03', 'CP04', 'CP05', 'CP06'
        ]

        self.cargasd: List[str] = [
            'CD01', 'CD02', 'CD03', 'CD04', 'CD05', 'CD06', 'CD08', 'CD09', 'CD10', 'CD11',
             'CD12', 'CD13', 'CD14', 'CD15', 'CD16', 'CD17', 'CD18', 'CD19'
        ]

        self.ramais: List[str] = [
            ' P1', ' P2', ' P3', ' P4', ' P5', ' P6', ' P7', ' P8'
        ] # espaço antes do nome para não confundir com os nomes das cargas

        self.cap: List[str] = [
            'CAP1'
        ] # CAP2 não está conectado atualmente

        # Carregamento e GD
        self.carregamento: List[float] = [1, 0.3]  # definir os carregamentos simulados
        self.Pgd: List[float] = [0.995, 0.665, 0.335]  # níveis de penetração da GD (ativa)
        self.Qgd: List[float] = [0.0995, 0.0665, 0.0335]  # 10% da ativa
        self.Spv: List[float] = [0.09, 0.06, 0.03]  # níveis de penetração da PV

        # Caminhos/linhas 

        self.parent_path: Path = Path(__file__).resolve().parent
        self.path: str = str(self.parent_path) + '\\SystemLines\\'
        #Contagem das linhas começa em zero, fazer -1 do arquivo de texto
        self.linhatsimu: int = 8          # linha onde se modifica o tempo de simulação
        self.linhamudasolo: int = 2100    # linha do sinal que entra no modelo ATP
        self.linhaschgd: int = 2877       # primeira linha (3 fases) da chave que desconecta a GD
        self.linhascap2: int = 3018       # primeira linha das chaves do capacitor 2
        self.linhaPGD: int = 11           # linha onde insere P (ativa) da GD
        self.linhaSPV: int = 15           # linha onde insere S da PV (novo)
        self.noI: int = 3289              # início do modelo FAI na barra 802
        self.noF: int = 2785              # fim do modelo FAI
        self.linhach: int = 3006          # linha entre o modelo e a chave do modelo da FAI
        self.linhachcargasI: int= 3011    # linha do arquivo em que começam as chaves que conectam as CD, CP, ramais e cap
        self.linhachcargasF: int= 3118    # linha do arquivo em que terminam as chaves que conectam as CD, CP, ramais e cap
        self.linhachPV: int = 14          # linha onde se conecta a PV
        self.linhachGDSinc: int = 3134    # linha onde se conecta a GD sincrona
        self.linhaparamFAI: int = 2096    # linha que inicia os paramtros da FAI, 20 parametros no total

        # Parâmetros do modelo FAI

        self.parametrosFAI: dict[str, str] = {
            'ruptt': 'TRUP',
            'toucht': 'TTOQ',
            'vfalta': '24900.',
            'ifalta': '5.',
            'soil': '0.0',
            'iperc': '1.',
            'cbup': '50.',
            'freq': '60.',
            'Iasy': '5.',
            'Ipasy': '10.',
            'ISpike': '60.',
            'IpSpike': '1.',
            'ModPos': '300.',
            'ModNeg': '10.',
            'mincy': '2.',
            'maxcy': '8.',
            'pModPos': '80.',
            'pModNeg': '10.',
            'decaimento': '500.',
            'randomizar': '1.'
        }

        # Cartão original usado como base
        self.cartao_original: str = str(self.parent_path) + '\\TestSystem\\Matheus_34barras_FAImodelo_ComMedidores_3GD_1GS_inversormedio.txt'
        # Dataframes de linhas a modificar (carregados no setup)
        self.Linhascd: pd.DataFrame | None = None
        self.Linhascp: pd.DataFrame | None = None
        self.LinhasBarras: pd.DataFrame | None = None

        # Cartões de trabalho
        self.CartaoA: List[List[str]] | None = None  # cópia do cartão original
        # Obs.: CartaoB/C/D/E são variáveis locais durante o loop, como no script

    # ======================================================================
    # Métodos auxiliares (apenas “organizam” o fluxo; não alteram a lógica)
    # ======================================================================

    def carregar_sistemas(self) -> None:
        """Carrega as planilhas de linhas do cartão do ATP que serão modificadas."""
        self.Linhascd = pd.read_excel(self.path + 'linhascargasdistribuidas.xlsx')
        self.Linhascp = pd.read_excel(self.path + 'linhascargaspontuais.xlsx')
        self.LinhasBarras = pd.read_excel(self.path + 'LinhasHIFReal.xlsx')
        self.LinhasParamFAI = pd.read_excel(self.path + 'linhasParametrosFAIadaptado.xlsx')

    def carregar_cartao_original(self) -> None:
        """Carrega o cartão ATP base (xlsx) e converte para lista de listas."""
        # CartaoATPOriginal = pd.read_excel(self.cartao_xlsx)
        CartaoATPOriginal = pd.read_csv(self.cartao_original,header=None, dtype=str, sep='\t', encoding='latin-1', skip_blank_lines=False).fillna('')
        self.CartaoA = CartaoATPOriginal.values.tolist()

    @staticmethod
    def selecionar_barras_por_fase(f: int,
                                   barrasfasea: List[int],
                                   barrasfaseb: List[int],
                                   barrasfasec: List[int]) -> List[int]:
        """Retorna a lista de barras conforme a fase f (1,2,3)."""
        if f == 1:
            return barrasfasea
        elif f == 2:
            return barrasfaseb
        elif f == 3:
            return barrasfasec
        # Mantém comportamento implícito (não ocorre no fluxo atual)
        return []

    @staticmethod
    def desconectar_gd_conectar_cap2(CartaoB: List[List[str]],
                                     linhaschgd: int,
                                     linhascap2: int) -> List[List[str]]:
        """
        Mantém a mesma sequência de substituições do código original para
        simular o sistema sem GD: ajusta 3 linhas consecutivas em chaves e capacitor.
        """
        for i in [0, 1, 2]:
            # Desconectando a GD
            linhamudar = CartaoB[linhaschgd + i]
            novalinha = [w.replace('-1.', '10.') for w in linhamudar]
            CartaoB[linhaschgd + i] = novalinha

            # Conectando o capacitor 2
            linhamudar = CartaoB[linhascap2 + i]
            novalinha = [w.replace('1.E3', ' -1.') for w in linhamudar]
            CartaoB[linhascap2 + i] = novalinha
        return CartaoB

    # =========================
    # Pipeline 
    # =========================
    def loop_completo(self) -> None:

        assert self.Linhascd is not None and self.Linhascp is not None and self.LinhasBarras is not None, \
            "Linhascd/Linhascp/LinhasBarras não carregadas. Chame carregar_sistemas()."
        assert self.CartaoA is not None, "Cartão base não carregado. Chame carregar_cartao_original()."

        CartaoA = self.CartaoA  # referência local

        # lista de fases e suas barras correspondentes (mantém exatamente o mesmo range)
        for f in [1,2,3]:  # range(1, len(Fases))  --> aqui só 1 e 2, como no código
            fase = self.Fases[f - 1]

            barrassimu = self.selecionar_barras_por_fase(
                f, self.barrasfasea, self.barrasfaseb, self.barrasfasec
            )

            for cgd in [1,2]:  # range(1,2) -> apenas com GD (1)
                CartaoB = CartaoA.copy()

                # Se quiser simular sem GD (cgd == 2)
                if cgd == 2:
                    CartaoB = desconecta_chaves(
                            CartaoA, 'TCHPV', self.linhachPV,
                            self.linhachPV + 1, True
                    ) # apenas 1 linha será alterada, +1 linha para permitir o espaço de busca do script

                    CartaoB = desconecta_chaves(
                            CartaoB, 'SWGD', self.linhachGDSinc,
                            self.linhachGDSinc + 3, False
                    ) # 3 linhas serão alteradas, +3 linha para permitir o espaço de busca do script

                # Nível de penetração da GD (usa só o primeiro: Pgd[:1])
                for pn, valor in enumerate(self.Pgd[:3], start=1):
                    CartaoC = CartaoB.copy()

                    # Se pn > 1, muda P/Q da GD (no fluxo atual, não entra)
                    if pn > 1:
                        CartaoC = muda_potencia_GD(
                            CartaoC,
                            self.linhaPGD,
                            self.Pgd,
                            self.Qgd,
                            pn,
                            '0.995',
                            '0.0995',
                            self.linhaSPV,
                            self.Spv,
                            '0.09'
                        ) #está ok

                    # Carregamento
                    for car in [1,2]:
                        CartaoD = CartaoC.copy()

                        if car > 1:
                            CartaoD = mudacarregamento(
                                CartaoC,
                                self.carregamento[car - 1],
                                self.Linhascd,
                                self.Linhascp,
                                self.path
                            )

                        # Próximas barras
                        # for pb in [1,2,3,4]:
                        for pb in range(1,29):
                            CartaoE = CartaoD.copy()
                            str_proxbarra = str(barrassimu[pb - 1])
                            proxbarra = barrassimu[pb - 1]
                            barra = 2 # a barra original é a 2, sempre muda da 2 para outra barra

                            # Para cada solo de FAI (usa somente 2: [2])
                            # Para FAI com modelo, vai modificar aqui a substituição apenas do solo, pela configuração do parametrizamodelo
                            for solos in [2]:
                                # CartaoE = faino34barras_HIFreal(
                                #     CartaoD, CartaoE, barra, proxbarra,
                                #     self.LinhasBarras, f,
                                #     self.noI, self.noF,
                                #     solos,
                                #     self.linhatsimu, self.linhamudasolo, self.linhach
                                # )

                                CartaoE = faino34barras_HIFmodelo2022(
                                    CartaoD, CartaoE, barra, proxbarra,
                                    self.LinhasBarras, f,
                                    self.noI, self.noF
                                )

                                # for carga in self.cargasp + self.cargasd + self.ramais + self.cap: # Concatena para iterar sobre todas as listas
                                for carga in ['nenhum']: # não retira nenhuma carga
                                    CartaoF = CartaoE.copy()
                                    
                                    # CartaoF = desconecta_chaves(
                                    #     CartaoE, carga, self.linhachcargasI,
                                    #     self.linhachcargasF, False
                                    # )
                                    
                                    
                                    for fai in self.LinhasParamFAI.columns[1:]: # itera sobre o número de colunas do arquivo de parâmetros FAI (linhasparamFAI)
                                        CartaoG = CartaoF.copy()
                                        CartaoG = parametrizamodelo(
                                            CartaoG, self.parametrosFAI, self.LinhasParamFAI[fai],
                                            self.linhaparamFAI
                                        )
                                       
                                        salvacartaonapasta(
                                            CartaoG,
                                            'atpalterado.atp',
                                            str_proxbarra,
                                            fase,
                                            cgd,
                                            pn,
                                            car,
                                            solos,
                                            carga.strip(),
                                            fai
                                        )


    # ----------------------------
    # Conveniência: executar tudo
    # ----------------------------
    def run(self) -> None:
        """Executa o setup e o loop completo."""
        t0 = time.time()
        self.carregar_sistemas()
        self.carregar_cartao_original()
        self.loop_completo()
        print(f"Concluído em {time.time() - t0:.2f}s.")


# ===========================================================
# Main (permite rodar direto: python este_arquivo.py)
# ===========================================================
if __name__ == "__main__":
    app = Automatiza34Barras()
    app.run()