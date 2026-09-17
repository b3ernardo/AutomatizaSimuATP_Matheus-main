# step3_Pl42Parquet.py
# -*- coding: utf-8 -*-
"""
Etapa 3: Lê arquivos .pl4 (gerados na etapa 2) e salva as medições em .parquet.

Novidades:
- Parâmetros para remover uma fatia de amostras ANTES do salvamento:
    N_remover = fs_por_ciclo * freq_rede_hz * tempo_remover_s
  com a opção de cortar do "inicio" ou do "fim".
- Suporte a filtros por padrão de nome usando filtro_variaveis.py

Mantido:
- Fluxo de processamento em lote ou unitário
- Salvamento completo (todas as variáveis) ou apenas variáveis de interesse
- Deduplicação de nomes é tratada no save_results_parquet (f_readpl4.py)

Requisitos:
- f_readpl4.py na mesma pasta, contendo readpl4(...) e save_results_parquet(...)
- filtro_variaveis.py na mesma pasta (opcional, apenas se usar filtros)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Iterable, Optional

import numpy as np

from f_readpl4 import readpl4, save_results_parquet

# Importação opcional do módulo de filtros
try:
    from f_selecionavariaveisATP import seleciona_variaveis
    FILTRO_DISPONIVEL = True
except ImportError:
    FILTRO_DISPONIVEL = False
    print("⚠️  Módulo filtro_variaveis.py não encontrado. Filtros por padrão não estarão disponíveis.")


class Pl4Processor:
    def __init__(
        self,
        pasta_entrada: str | os.PathLike = "step2_ArquivosPl4",
        pasta_saida: str | os.PathLike = "step3_ParquetOutput",
        modo_todos: bool = True,
        salvar_todas: bool = True,
        # ---- NOVOS CONTROLES DE CORTE ----
        remover_amostras: bool = False,     # habilita/desabilita o corte
        fs_por_ciclo: int = 128,            # amostras por ciclo
        freq_rede_hz: float = 60.0,         # frequência da rede
        tempo_remover_s: float = 0.0,       # segundos a remover
        remover_de: str = "inicio",         # "inicio" ou "fim"
        # ---- Seleção de variáveis de interesse (se salvar_todas=False) ----
        nomes_interesse: Optional[Iterable[str]] = None,  # ex.: ["IA", "IB", "IC", "VA", ...]
        # ---- NOVO: Filtros por padrão (alternativa a nomes_interesse) ----
        filtros_padrao: Optional[Iterable[str]] = None,  # ex.: ['N', 'GER', 'M_NUM', 'GS', 'SUB']
    ):
        """
        Processador de arquivos PL4 para Parquet.
        
        Parâmetros:
        -----------
        pasta_entrada : str ou Path
            Pasta contendo os arquivos .pl4
        pasta_saida : str ou Path
            Pasta onde serão salvos os arquivos .parquet
        modo_todos : bool
            Se True, processa todos os .pl4 da pasta_entrada
        salvar_todas : bool
            Se True, salva todas as variáveis; False: apenas as de interesse
        
        remover_amostras : bool
            Habilita o corte de amostras
        fs_por_ciclo : int
            Amostras por ciclo (ex: 128)
        freq_rede_hz : float
            Frequência da rede em Hz (ex: 60.0)
        tempo_remover_s : float
            Tempo em segundos a remover
        remover_de : str
            "inicio" ou "fim" - de onde remover as amostras
        
        nomes_interesse : list de str, opcional
            Lista explícita de nomes de variáveis (modo tradicional)
        
        filtros_padrao : list de str, opcional
            Lista de filtros para selecionar variáveis por padrão. Opções:
            - 'N': variáveis que começam com 'N' + número
            - 'GER': variáveis que começam com 'GER'
            - 'M_NUM' ou 'M': variáveis que começam com 'M' + número
            - 'GS': variáveis que começam com 'GS'
            - 'SUB': variáveis que começam com 'SUB'
            
            Exemplo: filtros_padrao=['N', 'GER', 'M_NUM']
            
            Nota: Se nomes_interesse for fornecido, tem prioridade sobre filtros_padrao
        """
        self.pasta_entrada = Path(pasta_entrada)
        self.pasta_saida = Path(pasta_saida)
        self.modo_todos = bool(modo_todos)
        self.salvar_todas = bool(salvar_todas)

        # parâmetros de corte
        self.remover_amostras = bool(remover_amostras)
        self.fs_por_ciclo = int(fs_por_ciclo)
        self.freq_rede_hz = float(freq_rede_hz)
        self.tempo_remover_s = float(tempo_remover_s)
        self.remover_de = str(remover_de).lower().strip()  # "inicio" | "fim"

        # seleção de variáveis
        self.nomes_interesse = list(nomes_interesse) if nomes_interesse else None
        self.filtros_padrao = list(filtros_padrao) if filtros_padrao else None
        
        # Validação
        if self.filtros_padrao and not FILTRO_DISPONIVEL:
            raise ImportError(
                "filtros_padrao requer o módulo filtro_variaveis.py. "
                "Certifique-se de que o arquivo está na mesma pasta."
            )

        self.pasta_saida.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Auxiliares de corte
    # -------------------------
    def _n_remover(self) -> int:
        """
        Fórmula pedida explicitamente:
            N = fs_por_ciclo * freq_rede_hz * tempo_remover_s
        Ex.: 128 * 60 * 1.0 = 7680
        """
        try:
            n = int(self.fs_por_ciclo * self.freq_rede_hz * self.tempo_remover_s)
        except Exception:
            n = 0
        return max(n, 0)

    def _recortar_resultado(self, resultado: Dict) -> Dict:
        """
        Aplica corte (se habilitado) na estrutura completa:
            resultado["time"] -> (N,)
            resultado["data"] -> (n_var, N)
        """
        if not (self.remover_amostras and self.tempo_remover_s > 0):
            return resultado

        time = np.asarray(resultado.get("time"))
        data = np.asarray(resultado.get("data"))
        if time.ndim != 1 or data.ndim != 2 or data.shape[1] != time.shape[0]:
            # Estruturas incompatíveis – não corta
            return resultado

        n_drop = self._n_remover()
        N = time.shape[0]
        if n_drop <= 0 or n_drop >= N:
            # nada a cortar, ou corte inválido
            return resultado

        if self.remover_de == "fim":
            time_new = time[:-n_drop]
            data_new = data[:, :-n_drop]
        else:
            time_new = time[n_drop:]
            data_new = data[:, n_drop:]

        res = dict(resultado)  # cópia rasa
        res["time"] = time_new
        res["data"] = data_new
        res["n_samples"] = int(time_new.shape[0])
        if "delta_t" in res:
            dt = float(res["delta_t"])
            res["tmax"] = (res["n_samples"] - 1) * dt
        return res

    def _recortar_medicoes(self, medicoes: Dict[str, np.ndarray | list | float | int]) -> Dict:
        """
        Aplica corte (se habilitado) em dicionário do tipo:
            {"time"/"tempo": (N,), "Var1": (N,), ...}
        Mantém valores que não tenham comprimento N.
        """
        if not (self.remover_amostras and self.tempo_remover_s > 0):
            return medicoes

        time_key = "time" if "time" in medicoes else ("tempo" if "tempo" in medicoes else None)
        if time_key is None:
            return medicoes

        t = np.asarray(medicoes[time_key])
        if t.ndim != 1:
            return medicoes

        N = t.shape[0]
        n_drop = self._n_remover()
        if n_drop <= 0 or n_drop >= N:
            return medicoes

        def corta(arr: np.ndarray) -> np.ndarray:
            if self.remover_de == "fim":
                return arr[:-n_drop]
            return arr[n_drop:]

        novo = {}
        for k, v in medicoes.items():
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.shape[0] == N:
                novo[k] = corta(arr)
            else:
                novo[k] = v
        return novo

    # -------------------------
    # Seleção de variáveis
    # -------------------------
    def selecionar_variaveis_interesse(self, resultado: Dict) -> Dict[str, np.ndarray]:
        """
        Constrói um dicionário {nome: série} contendo:
          - 'time' (sempre)
          - variáveis selecionadas por filtros ou nomes explícitos
        """
        # NOVO: Se filtros_padrao foi fornecido e nomes_interesse não foi
        if self.filtros_padrao and not self.nomes_interesse:
            if not FILTRO_DISPONIVEL:
                raise ImportError("filtro_variaveis.py não disponível")
            print(f"  🔍 Usando filtros por padrão: {self.filtros_padrao}")
            return seleciona_variaveis(
                resultado,
                filtros=self.filtros_padrao,
                incluir_time=True
            )
        
        # Modo tradicional (original)
        out: Dict[str, np.ndarray] = {}
        time = np.asarray(resultado["time"])
        data = np.asarray(resultado["data"])
        labels: List[str] = list(resultado.get("labels", []))

        out["time"] = time

        if self.nomes_interesse is None:
            # Se não foi passado um conjunto de nomes, retorna todas (modo "interesse" = todas)
            for i, lb in enumerate(labels):
                out[str(lb)] = data[i, :]
            return out

        # Seleciona apenas as pedidas
        nomes_set = {str(n) for n in self.nomes_interesse}
        for i, lb in enumerate(labels):
            nome = str(lb)
            if nome in nomes_set:
                out[nome] = data[i, :]
        return out

    # -------------------------
    # Processamento
    # -------------------------
    def processar_um(self, caminho_arquivo: str | os.PathLike):
        """
        Processa um único .pl4 -> salva .parquet na pasta de saída.
        """
        caminho_arquivo = Path(caminho_arquivo)
        nome_base = caminho_arquivo.stem
        print(f"\n📂 Lendo arquivo: {caminho_arquivo}")

        resultado = readpl4(str(caminho_arquivo))

        print("\n📘 Resumo (antes do corte):")
        if "labels" in resultado:
            print(f"  • Variáveis: {len(resultado['labels'])}")
        if "time" in resultado and "data" in resultado:
            print(f"  • Amostras: {np.asarray(resultado['time']).shape[0]}")

        # --- CORTE OPCIONAL (estrutura completa) ---
        if self.remover_amostras and self.tempo_remover_s > 0:
            resultado = self._recortar_resultado(resultado)

        print("\n📘 Resumo (após o corte):")
        if "labels" in resultado:
            print(f"  • Variáveis: {len(resultado['labels'])}")
        if "time" in resultado and "data" in resultado:
            print(f"  • Amostras: {np.asarray(resultado['time']).shape[0]}")

        # Caminho de saída
        saida_parquet = self.pasta_saida / f"{nome_base}.parquet"

        # Salvar todas as variáveis ou somente as de interesse
        if self.salvar_todas:
            path, nlin, cols = save_results_parquet(
                str(saida_parquet),
                resultado,
                usar_variaveis_interesse=False,
                # repassa também os parâmetros de corte (idempotente aqui)
                remover_amostras=self.remover_amostras,
                fs_por_ciclo=self.fs_por_ciclo,
                freq_rede_hz=self.freq_rede_hz,
                tempo_remover_s=self.tempo_remover_s,
                remover_de=self.remover_de,
            )
        else:
            medicoes = self.selecionar_variaveis_interesse(resultado)

            # --- CORTE OPCIONAL (dicionário de interesse) ---
            if self.remover_amostras and self.tempo_remover_s > 0:
                medicoes = self._recortar_medicoes(medicoes)

            path, nlin, cols = save_results_parquet(
                str(saida_parquet),
                medicoes,
                usar_variaveis_interesse=True,
                remover_amostras=self.remover_amostras,
                fs_por_ciclo=self.fs_por_ciclo,
                freq_rede_hz=self.freq_rede_hz,
                tempo_remover_s=self.tempo_remover_s,
                remover_de=self.remover_de,
            )

        print(f"\n✅ Salvo: {path}")
        print(f"   Linhas: {nlin}")
        print(f"   Colunas: {len(cols)}")

    def executar(self):
        """
        Executa o processamento em lote (todos .pl4) ou solicita um único arquivo.
        """
        if self.modo_todos:
            pl4s = sorted(self.pasta_entrada.glob("*.pl4"))
            if not pl4s:
                print(f"⚠️  Nenhum .pl4 encontrado em: {self.pasta_entrada.resolve()}")
                return
            print(f"▶️  Processando {len(pl4s)} arquivo(s) .pl4 de {self.pasta_entrada.resolve()}")
            for p in pl4s:
                try:
                    self.processar_um(p)
                except Exception as e:
                    print(f"❌ Falha em {p}: {e}")
        else:
            print("Defina modo_todos=True ou chame processar_um(caminho) diretamente.")


# -----------------------------------------------------------------------------
# MAIN (exemplo de uso)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Ajuste os parâmetros abaixo conforme sua necessidade.
    Se quiser desativar o corte, coloque remover_amostras=False ou tempo_remover_s=0.0.
    """
    SalvarTodasasvariaveis = False # se false, salva as filtradas
    
    if SalvarTodasasvariaveis == True:
        # =========================================================================
        # EXEMPLO 1: Salvar todas as variáveis (modo original)
        # =========================================================================
        processor = Pl4Processor(
            pasta_entrada="step2_ArquivosPl4",
            pasta_saida="step3_ParquetOutput",
            modo_todos=True,        # True: processa todos .pl4 da pasta_entrada
            salvar_todas=True,      # True: salva todas as variáveis
            # ---- CONTROLES DE CORTE ----
            remover_amostras=True,  # habilita o corte antes do salvamento
            fs_por_ciclo=128,       # amostras por ciclo (ex.: 128)
            freq_rede_hz=60.0,      # frequência da rede (ex.: 60 Hz)
            tempo_remover_s=1.0,    # segundos a remover (ex.: 1.0 s -> 128*60*1 = 7680 amostras)
            remover_de="inicio",    # "inicio" ou "fim"
        )
        processor.executar()
    else:    
        # =========================================================================
        # EXEMPLO 2: Salvar apenas variáveis de interesse com FILTROS (NOVO!)
        # =========================================================================
        processor = Pl4Processor(
             pasta_entrada="step2_ArquivosPl4",
             pasta_saida="step3_ParquetOutput_Filtrado",
             modo_todos=True,
             salvar_todas=False,     # False: apenas variáveis de interesse
             # ---- CONTROLES DE CORTE ----
             remover_amostras=True,
             fs_por_ciclo=128,
             freq_rede_hz=60.0,
             tempo_remover_s=1.0,
             remover_de="inicio",
             # ---- NOVO: FILTROS POR PADRÃO ----
             filtros_padrao=['N', 'GER', 'M_NUM', 'GS', 'SUB'],  # Seleciona por padrão
         )
        processor.executar()
    
    # =========================================================================
    # EXEMPLO 3: Salvar variáveis específicas por nome (modo tradicional)
    # =========================================================================
    # processor = Pl4Processor(
    #     pasta_entrada="step2_ArquivosPl4",
    #     pasta_saida="step3_ParquetOutput_Especificas",
    #     modo_todos=True,
    #     salvar_todas=False,
    #     remover_amostras=False,  # sem corte
    #     # ---- NOMES ESPECÍFICOS ----
    #     nomes_interesse=["SUBA- (V-node)", "SUBB- (V-node)", "SUBC- (V-node)", 
    #                      "GERA-SUBA (I-bran)", "GERB-SUBB (I-bran)", "GERC-SUBC (I-bran)"],
    # )
    # processor.executar()