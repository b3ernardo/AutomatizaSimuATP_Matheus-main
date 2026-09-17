# -*- coding: utf-8 -*-
"""
Lib: Automatização 34 Barras (organizado)
- Mantém a lógica original das funções.
- Importações consolidadas no topo.
- Helpers para evitar repetição de código de formatação/substituição.

Criado a partir do arquivo do usuário (organização/limpeza).
"""

from pathlib import Path
from typing import List, Tuple, Sequence, Union, Iterable, Optional

import numpy as np
import pandas as pd
import scipy.io  # usado em salvaoarquivomatlab
import h5py      # usado em salvaoarquivomatlab

# Tipagens auxiliares
StrList = List[str]
Line = List[str]
CartaoType = List[Line]
Matrix = Sequence[Sequence[Union[int, float]]]

# =========================================================
# Helpers (não alteram a lógica; apenas reduzem repetição)
# =========================================================

def _two_digits(n: int) -> str:
    """Formata inteiro com 2 dígitos (ex.: 3 -> '03')."""
    return f"{int(n):02d}"

def _idx(i: int, one_based: bool) -> int:
    """Converte índice 1-based para 0-based quando necessário."""
    return int(i) - 1 if one_based else int(i)

def _replace_line_tokens(line: Line, old: str, new: str) -> Line:
    """Aplica replace token a token (mantém padrão do código original)."""
    return [w.replace(old, new) for w in line]

def _pad_same_len(a: str, b: str) -> Tuple[str, str]:
    """
    Ajusta strings com espaços à esquerda para ficarem do mesmo comprimento,
    preservando a lógica de pad usada no código original.
    """
    if len(a) > len(b):
        b = " " * (len(a) - len(b)) + b
    elif len(b) > len(a):
        a = " " * (len(b) - len(a)) + a
    return a, b

def _format_to_exp_token(token: str) -> str:
    """
    Replica a formatação ad-hoc do código original:
    - Recebe algo como '1234.' (str), usa len(token) para definir 'x.yEz'.
    - Mantém a mesma heurística de tamanhos do código original.
    """
    z = token
    if len(z) == 8:
        # ex.: '1234.567' -> '1.3E6' seguindo a lógica original
        z = z[0] + "." + z[2] + "E6"
    elif len(z) == 7:
        z = z[0] + "." + z[2] + "E5"
    return z

def _safe_get_column(df: pd.DataFrame, prefer: str, fallback: Optional[str] = None) -> List[Union[int, float]]:
    """
    Lê coluna preferida; se não existir, tenta fallback.
    Mantém comportamento original, mas torna o acesso resiliente.
    """
    if prefer in df.columns:
        return df[prefer].values.tolist()
    if fallback and fallback in df.columns:
        return df[fallback].values.tolist()
    # se nenhuma existir, lança erro claro
    raise KeyError(f"Coluna '{prefer}' (ou fallback '{fallback}') não encontrada no DataFrame.")

# =========================================================
# Funções (lógica original preservada)
# =========================================================

def mudacarregamento(Cartao: CartaoType, car: float,
                     Linhascd: dict, Linhascp: dict, path: str) -> CartaoType:
    """
    Atualiza impedâncias de cargas distribuídas e pontuais no cartão ATP.

    Params:
      Cartao: List[List[str]]
      car: fator de escala do carregamento
      Linhascd: dict com chaves 'R' e 'L' (linhas das cargas distribuídas)
      Linhascp: dict com chaves 'R' e 'L' (linhas das cargas pontuais)
      path: pasta onde estão os .xlsx de impedâncias

    Retorna:
      NovoCartao (List[List[str]])
    """
    NovoCartao = Cartao

    Zcd, Zcp = [], []
    ZcdO, ZcpO = [], []
    Rdist, Xdist, Rpont, Xpont = [], [], [], []

    # Linhas do cartão com as impedâncias de cargas distribuídas
    matriz = pd.read_excel(Path(path) / 'ImpedanciaCargasDistribuidas.xlsx')
    RdistO = _safe_get_column(matriz, "Rdist")
    XdistO = _safe_get_column(matriz, "Xdist")

    matriz = pd.read_excel(Path(path) / 'ImpedanciaCargasPontuais.xlsx')
    RpontO = _safe_get_column(matriz, "Rpont")
    # Obs.: havia um provável typo no arquivo original usando "Rpont" para Xpont.
    # Mantemos a lógica, mas tentamos "Xpont" e, em fallback, repetimos "Rpont"
    XpontO = _safe_get_column(matriz, "Xpont", fallback="Rpont")

    ZcdO.append(RdistO)
    ZcdO.append(XdistO)
    ZcpO.append(RpontO)
    ZcpO.append(XpontO)

    # Calculando o novo valor das impedâncias (mesma lógica: round(x/car))
    Rdist[:] = [round(x / car) for x in RdistO]
    Xdist[:] = [round(x / car) for x in XdistO]
    Rpont[:] = [round(x / car) for x in RpontO]
    Xpont[:] = [round(x / car) for x in XpontO]

    Zcd.append(Rdist)
    Zcd.append(Xdist)
    Zcp.append(Rpont)
    Zcp.append(Xpont)

    # --- Substituição das cargas distribuídas
    for tc in [0, 1]:  # 0: R, 1: L (coluna)
        # for cargas in range(0, len(Zcd) - 1): #executa apenas a primeira carga
        for cargas in range(0, len(Zcd[tc])): #executa todas as cargas
            Ztc = Zcd[tc]
            Znovo = f"{Ztc[cargas]}."
            Ztc_orig = ZcdO[tc]
            Zantigo = f"{Ztc_orig[cargas]}."

            # formatação "exponencial" idêntica à original, apenas formata a string
            Znovo_fmt = _format_to_exp_token(Znovo)
            # padding para manter comprimentos
            Zantigo, Znovo_fmt = _pad_same_len(Zantigo, Znovo_fmt)

            # Linhas a modificar
            if tc == 0:
                lcd = Linhascd["R"][cargas]
            else:
                lcd = Linhascd["L"][cargas]

            # Substituição
            linhamudar = Cartao[lcd]
            novalinha = _replace_line_tokens(linhamudar, Zantigo, Znovo_fmt)
            NovoCartao[lcd] = novalinha

            if len(Zantigo) != len(Znovo_fmt):
                print('problema')
                break

    # --- Substituição das cargas pontuais
    for tc in [0, 1]:  # 0: R, 1: L
        for cargas in range(0, len(Zcp[tc])):
            Ztc = Zcp[tc]
            Znovo = f"{Ztc[cargas]}."
            Ztc_orig = ZcpO[tc]
            Zantigo = f"{Ztc_orig[cargas]}."

            Znovo_fmt = _format_to_exp_token(Znovo)
            Zantigo, Znovo_fmt = _pad_same_len(Zantigo, Znovo_fmt)

            if tc == 0:
                lcd = Linhascp["R"][cargas]
            else:
                lcd = Linhascp["L"][cargas]

            linhamudar = NovoCartao[lcd]
            novalinha = _replace_line_tokens(linhamudar, Zantigo, Znovo_fmt)
            NovoCartao[lcd] = novalinha

            if len(Zantigo) != len(Znovo_fmt):
                print('problema')
                break

    return NovoCartao


def salvacartaonapasta(Cartao: CartaoType, cardNameatpalterado: str,
                       str_proxbarra: str, fase: str, cgd: int, pn: int,
                       car: float, solos: int, carga: str, fai: str) -> int:
    """
    Salva o cartão em árvore de pastas específica (com/sem GD).
    Mantém a estrutura de nomes e o recorte Cartao[3:].
    """
    # Converte o cartão para DataFrame (mantém como no original)
    # CartaoN = Cartao[3:] #não fez sentido retirar as linhas
    CartaoN = Cartao

    CartaoDF = pd.DataFrame(CartaoN, columns=['dados'])

    # Monta o caminho de saída
    # if cgd == 1:
    #     nomeSinal = f"p{pn}/carregamento{car}/barra{str_proxbarra}solo{solos}retirado{carga}"
    #     saida = Path("CartoesATP") / fase / "comGD" / nomeSinal
    # else:
    #     nomeSinal = f"carregamento{car}/barra{str_proxbarra}solo{solos}retirado{carga}"
    #     saida = Path("CartoesATP") / fase / "semGD" / nomeSinal

    # Todos arquivos salvos na mesma pasta
    if cgd == 1:
        nomeSinal = f"{fase}_comGD_p{pn}_carregamento{car}_barra{str_proxbarra}solo{solos}retirado{carga}fai{fai}"
        saida = Path("CartoesATP") / nomeSinal
    else:
        nomeSinal = f"{fase}_semGD_p{pn}_carregamento{car}_barra{str_proxbarra}solo{solos}retirado{carga}fai{fai}"
        saida = Path("CartoesATP") / nomeSinal

    # Caminho completo do arquivo final
    nomeDoCartao = saida.with_suffix(".atp")

    # Cria apenas as pastas (não inclui o arquivo)
    nomeDoCartao.parent.mkdir(parents=True, exist_ok=True)
    print(f"Salvando cartão em: {nomeDoCartao}")
    # Salva o cartão
    np.savetxt(nomeDoCartao, CartaoDF.values, fmt='%s')
    return 0


def muda_potencia_GD(Cartao: CartaoType, linhaPGD: int,
                     Pgd: List[float], Qgd: List[float],
                     pn: int, Poriginal: str, Qoriginal: str, linhaSPV:int, 
                     Spv: List[float], Spvoriginal: str) -> CartaoType:
    """
    Atualiza as potências ativa e reativa (GD) nas linhas do cartão ATP.
    Lógica preservada: substitui tokens na linhaPGD e linhaPGD+1.
    """
    # potência ativa
    linhamudar = Cartao[linhaPGD]
    # print('cartao antes:', Cartao[linhaPGD])
    # print('linhamudar:', linhamudar)
    novalinha = _replace_line_tokens(linhamudar, Poriginal, str(Pgd[pn - 1]))
    # print('pgd:', Pgd[pn - 1])
    # print('novalinha:', novalinha)
    Cartao[linhaPGD] = novalinha

    # potência reativa
    linhamudar = Cartao[linhaPGD + 1]
    novalinha = _replace_line_tokens(linhamudar, Qoriginal, str(Qgd[pn - 1]))
    Cartao[linhaPGD + 1] = novalinha

    # potência aparente PV
    linhamudar = Cartao[linhaSPV]
    novalinha = _replace_line_tokens(linhamudar, Spvoriginal, str(Spv[pn - 1]))
    Cartao[linhaSPV] = novalinha

    return Cartao


def faino34barras_HIFreal(CartaoA: CartaoType, CartaoB: CartaoType,
                          barra: int, proxbarra: int,
                          LinhasBarras: pd.DataFrame, fs: int,
                          noI: int, noF: int, solos: int,
                          linhatsimu: int, linhamudasolo: int, linhach: int) -> CartaoType:
    """
    Implementação 'HIFreal' (como no arquivo original), mantendo a lógica.
    """
    str_barra = _two_digits(barra)
    str_proxbarra = _two_digits(proxbarra)

    linhaoriginali = noI
    linhaoriginalf = noF

    # LinhasBarras(proxbarra, fs) no MATLAB -> aqui a versão original
    LB = LinhasBarras.values.tolist()
    LinhaEspecifica = LB[proxbarra - 1] #linha da lista
    Linhaproxima = int(LinhaEspecifica[fs]) #apenas o número da linha do arquivo atp

    # nome da fase
    if fs == 1:
        nfase = 'A'
    elif fs == 2:
        nfase = 'B'
    else:
        nfase = 'C'

    # Barras monofásicas
    if proxbarra in [9, 13, 18, 30, 10, 12, 15, 23]:
        # a) resistência -> N02
        linhamudar = CartaoA[Linhaproxima]
        novalinha = _replace_line_tokens(linhamudar, f"N{str_proxbarra}I ", f"N{str_barra}I{nfase}")
        CartaoB[Linhaproxima] = novalinha

        linhamudar = CartaoA[Linhaproxima]
        novalinha = _replace_line_tokens(linhamudar, f"N{str_proxbarra}F ", f"N{str_barra}F{nfase}")
        CartaoB[Linhaproxima] = novalinha

        # b) transfere modelo da N02 para a barra
        linhamudar = CartaoA[linhaoriginali]
        novalinha = _replace_line_tokens(linhamudar, f"N{str_barra}I{nfase}", f"N{str_proxbarra}I ")
        CartaoB[linhaoriginali] = novalinha

        linhamudar = CartaoA[linhaoriginalf]
        novalinha = _replace_line_tokens(linhamudar, f"N{str_barra}F{nfase}", f"N{str_proxbarra}F ")
        CartaoB[linhaoriginalf] = novalinha

        # c) linha da chave - VERIFICAR SE PRECISA
        # linhamudar = CartaoA[linhach]
        # novalinha = _replace_line_tokens(linhamudar, f"N{str_barra}I{nfase}", f"N{str_proxbarra}I ")
        # CartaoB[linhach] = novalinha

    else:
        # Barras trifásicas
        linhamudar = CartaoA[Linhaproxima]
        novalinha = _replace_line_tokens(linhamudar, f"N{str_proxbarra}", f"N{str_barra}")
        CartaoB[Linhaproxima] = novalinha

        linhamudar = CartaoA[linhaoriginalf]
        novalinha = _replace_line_tokens(linhamudar, f"N{str_barra}", f"N{str_proxbarra}")
        CartaoB[linhaoriginalf] = novalinha

        linhamudar = CartaoA[linhaoriginali]
        novalinha = _replace_line_tokens(linhamudar, str_barra, str_proxbarra)
        CartaoB[linhaoriginali] = novalinha

        # linhamudar = CartaoA[linhach]
        # novalinha = _replace_line_tokens(linhamudar, f"N{str_barra}I{nfase}", f"N{str_proxbarra}I{nfase}")
        # CartaoB[linhach] = novalinha

    # # temposimu original
    # temposimu = [356, 455, 686, 659, 964, 770, 650, 612, 810, 534, 880, 946, 894,
    #              517, 290, 394, 876, 486, 789, 906, 788, 728, 679, 675, 381, 284,
    #              380, 621, 527, 509, 646, 483, 609, 624]
    # tsimu = temposimu[solos - 1] - 60
    # if tsimu > 390:
    #     tsimu = 390

    # linhamudar = CartaoA[linhatsimu]
    # novalinha = _replace_line_tokens(linhamudar, '300', str(tsimu))
    # CartaoB[linhatsimu] = novalinha

    # # Mudança de sinal (FAI)
    # linhamudar = CartaoA[linhamudasolo]
    # novalinha = _replace_line_tokens(linhamudar, '1.txt', f'{solos}.txt')
    # CartaoB[linhamudasolo] = novalinha

    return CartaoB


def salvaoarquivomatlab(PastaSalvar: str, str_proxbarra: str,
                        solos: int, barrastrimono: Iterable[int]) -> int:
    """
    Esqueleto de leitura/salvamento .mat (7.3) — preserva o comportamento esboçado.
    """
    folder = 'G:/Meu Drive/DOUTORADO/simulacoesATP_2023/HIFReal_Python'
    nome = Path(folder) / f'barra{str_proxbarra}solo{solos}.mat'

    # Exemplo de leituras (mantido como no original)
    mat = scipy.io.loadmat(str(nome))  # noqa: F841
    f = h5py.File(str(nome), 'r')      # noqa: F841

    # TODO: ler pl4 e salvar variáveis conforme sua pipeline
    return 0


def faino34barras_HIFmodelo2022(CartaoA: CartaoType, CartaoB: CartaoType,
                          barra: int, proxbarra: int,
                          LinhasBarras: pd.DataFrame, fs: int,
                          noI: int, noF: int) -> CartaoType:
    """
    Replica a função MATLAB faino34barras_HIFmodelo2022 (port Python).
    Mantém a lógica, apenas organiza e tipa.
    """
    
    str_barra = _two_digits(barra)
    str_proxbarra = _two_digits(proxbarra)

    linhaoriginali = noI
    linhaoriginalf = noF

    LB = LinhasBarras.values.tolist()
    LinhaEspecifica = LB[proxbarra - 1] #linha da lista
    Linhaproxima = int(LinhaEspecifica[fs]) #apenas o número da linha do arquivo atp

    # nome da fase
    if fs == 1:
        nfase = 'A'
    elif fs == 2:
        nfase = 'B'
    else:
        nfase = 'C'

    # Barras monofásicas
    if proxbarra in [9, 13, 18, 30, 10, 12, 15, 23]:
        # a) resistência -> N02
        linhamudar = CartaoA[Linhaproxima]
        novalinha = _replace_line_tokens(linhamudar, f"N{str_proxbarra}I ", f"N{str_barra}I{nfase}")
        novalinha = _replace_line_tokens(novalinha, f"N{str_proxbarra}F ", f"N{str_barra}F{nfase}")
        CartaoB[Linhaproxima] = novalinha

        # linhamudar = CartaoA[Linhaproxima]
        # novalinha = _replace_line_tokens(linhamudar, f"N{str_proxbarra}F ", f"N{str_barra}F{nfase}")
        # CartaoB[Linhaproxima] = novalinha

        # b) transfere modelo da N02 para a barra
        linhamudar = CartaoA[linhaoriginali]
        novalinha = _replace_line_tokens(linhamudar, f"N{str_barra}I{nfase}", f"N{str_proxbarra}I ")
        CartaoB[linhaoriginali] = novalinha

        linhamudar = CartaoA[linhaoriginalf]
        novalinha = _replace_line_tokens(linhamudar, f"N{str_barra}F{nfase}", f"N{str_proxbarra}F ")
        CartaoB[linhaoriginalf] = novalinha

        # c) linha da chave - VERIFICAR SE PRECISA
        # linhamudar = CartaoA[linhach]
        # novalinha = _replace_line_tokens(linhamudar, f"N{str_barra}I{nfase}", f"N{str_proxbarra}I ")
        # CartaoB[linhach] = novalinha

    else:
        # Barras trifásicas
        linhamudar = CartaoA[Linhaproxima]
        novalinha = _replace_line_tokens(linhamudar, f"N{str_proxbarra}F{nfase}", f"N{str_barra}FA")
        novalinha = _replace_line_tokens(novalinha, f"N{str_proxbarra}I{nfase}", f"N{str_barra}IA")
        CartaoB[Linhaproxima] = novalinha

        linhamudar = CartaoA[linhaoriginalf]
        novalinha = _replace_line_tokens(linhamudar, f"N{str_barra}FA", f"N{str_proxbarra}F{nfase}")
        CartaoB[linhaoriginalf] = novalinha

        linhamudar = CartaoA[linhaoriginali]
        novalinha = _replace_line_tokens(linhamudar, f"N{str_barra}IA", f"N{str_proxbarra}I{nfase}")
        CartaoB[linhaoriginali] = novalinha

    return CartaoB


# def parametrizamodelo(E: List[str],
#                       Parametros: List[str],
#                       ParametrosIniciais: List[str],
#                       LinhasModeloFAI: List[int]) -> List[str]:
#     """
#     Args:
#         E: lista de strings (cada elemento é uma linha do cartão ATP)
#         Parametros: lista de novos parâmetros (strings)
#         ParametrosIniciais: lista de parâmetros antigos (strings)
#         LinhasModeloFAI: lista de índices (1-based, como no MATLAB)

#     Returns:
#         E (lista modificada)
#     """

#     for pr in range(len(Parametros)):
#         linha = LinhasModeloFAI[pr]  # linha 1-based
#         str_antes = str(ParametrosIniciais[pr])
#         str_novo = str(Parametros[pr])

#         # padding à esquerda com espaços até 9 caracteres (como no MATLAB)
#         nulos = 9 - len(str_antes)
#         str_antesTOTAL = " " * nulos + str_antes
#         nulos = 9 - len(str_novo)
#         str_novoTOTAL = " " * nulos + str_novo

#         # substitui na linha correspondente (convertendo índice para 0-based)
#         E[linha - 1] = E[linha - 1].replace(str_antesTOTAL, str_novoTOTAL)

#     return E

def desconecta_chaves(Cartao: CartaoType, carga: str,
                     linhaI: int, linhaF: int, ponto: bool) -> CartaoType:
    """
    Altera o tempo de fechamento da chave de "carga" atual para que fique desconectado durante
    a simulação. "linhaI" e "linhaF" indicam as linhas no arquivo de início e fim da seção que tem 
    as chaves das cargas. "ponto" indica se deve ou não ser adicionado um ponto decimal ao número, necessário 
    para a seção "$PARAMETER".
    """
    
    linhas_encontradas = []
    
    # Garante que a iteração não ultrapasse o número total de linhas do arquivo
    limite_final = min(linhaF, len(Cartao))
    
    for i in range(linhaI, limite_final):
        if carga in Cartao[i][0]:
            linhas_encontradas.append(i)     
    
    if ponto:
        for i in linhas_encontradas:
            linhamudar = Cartao[i]
            novalinha = _replace_line_tokens(linhamudar, "-1. ", "100.") # Transfere o fechamento da chave para o instante 100s.
            Cartao[i] = novalinha
    else:
        for i in linhas_encontradas:
            linhamudar = Cartao[i]
            novalinha = _replace_line_tokens(linhamudar, "-1.", "100") # Transfere o fechamento da chave para o instante 100s.
            Cartao[i] = novalinha

    return Cartao

def parametrizamodelo(Cartao: CartaoType, parametrosini: dict, parametrosnovos: List[str],
                     linhaI: int) -> CartaoType:
    """
    Recebe o cartão ATP, um dicionário de parâmetros iniciais (chave: nome do parâmetro, valor: valor inicial),
    uma lista de novos valores de parâmetros (na mesma ordem que os parâmetros iniciais) e o índice da linha 
    inicial onde os parâmetros estão localizados.
    """
    
    lista_ini = list(parametrosini.items())
    
    for i, valor_novo in enumerate(parametrosnovos):
        # print('i: ', i, ' valor_novo: ', valor_novo)
        linha_atual = linhaI + i
        
        # Proteção para não estourar o limite de linhas do arquivo
        if linha_atual >= len(Cartao):
            break
        
        _, valor_antigo = lista_ini[i]
        # print('linha_atual: ', linha_atual, ' valor_antigo: ', valor_antigo, ' valor_novo: ', valor_novo)
        
        # Substitui a ocorrência do valor antigo pelo novo na linha correspondente
        # print('Cartao[linha_atual]: ', Cartao[linha_atual])
        Cartao[linha_atual] = _replace_line_tokens(Cartao[linha_atual], str(valor_antigo), str(valor_novo))
        
    return Cartao