# -*- coding: utf-8 -*-
"""
Conversão em POO para leitura de arquivos PL4 e salvamento em .pkl
Mantém o funcionamento original, adicionando opção de salvar apenas variáveis de interesse.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from f_readpl4 import readpl4, save_results


# =============================================================================
# FUNÇÃO AUXILIAR DE BUSCA
# =============================================================================
def busca_var(df, nome):
    """Retorna a coluna correspondente ao nome do nó (exato ou parcial)."""
    for col in df.columns:
        if nome in col:
            return df[col].values
    print(f"⚠️ Variável {nome} não encontrada.")
    return np.zeros_like(df["time"].values)


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================
class Pl4Processor:
    def __init__(
        self,
        pasta_entrada="ArquivoPl4",
        pasta_saida="Output",
        modo_todos=False,
        salvar_todas=True
    ):
        """
        Inicializa o processador de arquivos PL4.

        :param pasta_entrada: Pasta contendo arquivos .pl4
        :param pasta_saida: Pasta onde os .pkl serão salvos
        :param modo_todos: Se True, processa todos os arquivos da pasta; senão, apenas o primeiro
        :param salvar_todas: Se True, salva todas as variáveis. Se False, salva apenas variáveis de interesse
        """
        self.pasta_entrada = pasta_entrada
        self.pasta_saida = pasta_saida
        self.modo_todos = modo_todos
        self.salvar_todas = salvar_todas

        os.makedirs(self.pasta_saida, exist_ok=True)

    # -------------------------------------------------------------------------
    def listar_arquivos(self):
        """Retorna lista de arquivos .pl4 na pasta de entrada"""
        pl4_files = [
            os.path.join(self.pasta_entrada, f)
            for f in os.listdir(self.pasta_entrada)
            if f.lower().endswith(".pl4")
        ]
        pl4_files.sort()
        return pl4_files

    # -------------------------------------------------------------------------
    def selecionar_variaveis_interesse(self, resultado):
        """Filtra apenas as variáveis de interesse (V1a..I2c)"""
        time = resultado["time"]
        data = resultado["data"]
        varinfo = resultado["var_info"]

        # Cria DataFrame para facilitar busca
        df = pd.DataFrame(data.T, columns=[f"{v['FROM']}-{v['TO']} ({v['TYPE_NAME']})" for v in varinfo])
        df.insert(0, "time", time)

        # Tensões (V-node)
        V1a = busca_var(df, "BUZ1A")
        V1b = busca_var(df, "BUZ1B")
        V1c = busca_var(df, "BUZ1C")
        V2a = busca_var(df, "BUZ2A")
        V2b = busca_var(df, "BUZ2B")
        V2c = busca_var(df, "BUZ2C")

        # Correntes (I-bran)
        I1a = busca_var(df, "BUZ1A-BUZSA")
        I1b = busca_var(df, "BUZ1B-BUZSB")
        I1c = busca_var(df, "BUZ1C-BUZSC")
        I2a = busca_var(df, "BUZ2A-BUZ2SA")
        I2b = busca_var(df, "BUZ2B-BUZ2SB")
        I2c = busca_var(df, "BUZ2C-BUZ2SC")

        medicoes = {
            "tempo": time,
            "V1a": V1a, "V1b": V1b, "V1c": V1c,
            "V2a": V2a, "V2b": V2b, "V2c": V2c,
            "I1a": I1a, "I1b": I1b, "I1c": I1c,
            "I2a": I2a, "I2b": I2b, "I2c": I2c
        }

        return medicoes

    # -------------------------------------------------------------------------
    def salvar_variaveis_interesse(self, medicoes, base_out):
        """Salva as variáveis de interesse em formato .pkl"""
        import pickle
        out_path = f"{base_out}_interesse.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(medicoes, f)
        print(f"✅ Variáveis de interesse salvas em {out_path}")

    # -------------------------------------------------------------------------
    def processar_um(self, caminho_arquivo):
        """Processa um único arquivo PL4"""
        nome_base = os.path.splitext(os.path.basename(caminho_arquivo))[0]
        print(f"\n🔍 Lendo arquivo: {caminho_arquivo}")

        resultado = readpl4(caminho_arquivo)

        print("\n📘 Resumo:")
        print(f"  Método    : {resultado['method']}")
        print(f"  Δt        : {resultado['delta_t']:.6e} s")
        print(f"  tmax      : {resultado['tmax']:.4f} s")
        print(f"  Canais    : {resultado['n_channels']}")
        print(f"  Amostras  : {resultado['n_samples']}")

        base_out = os.path.join(self.pasta_saida, nome_base)

        # =================== ESCOLHA DO USUÁRIO ===================
        if self.salvar_todas:
            print("\n💾 Salvando TODAS as variáveis (modo completo)...")
            save_results(resultado, self.pasta_saida, base_name=nome_base, approach_num=8)
        else:
            print("\n💾 Salvando apenas variáveis de interesse...")
            medicoes = self.selecionar_variaveis_interesse(resultado)
            self.salvar_variaveis_interesse(medicoes, base_out)
        # ==========================================================

        # Plotagem rápida (mantém funcionalidade original)
        tempo = resultado["time"]
        dados = resultado["data"]
        labels = resultado["labels"]

        plt.figure(figsize=(12, 6))
        for i in range(min(6, len(labels))):
            plt.plot(tempo, dados[i, :], label=labels[i])
        '''
        plt.xlabel("Tempo (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Formas de onda - {nome_base}")
        plt.legend(loc="best", fontsize="x-small")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        '''
        print(f"✅ Arquivo {nome_base} processado com sucesso.")

    # -------------------------------------------------------------------------
    def executar(self):
        """Executa o processamento conforme o modo selecionado"""
        arquivos = self.listar_arquivos()
        if not arquivos:
            print("⚠️ Nenhum arquivo .pl4 encontrado em:", self.pasta_entrada)
            return

        print(f"📂 {len(arquivos)} arquivo(s) encontrado(s).")
        if self.modo_todos:
            print("🔁 Modo: processar TODOS os arquivos .pl4")
            for arq in arquivos:
                self.processar_um(arq)
        else:
            print("▶️ Modo: processar apenas o PRIMEIRO arquivo")
            self.processar_um(arquivos[0])

        print("\n✅ Processamento concluído.")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Configurações do usuário
    PROCESSAR_TODOS = True    # True = processar todos / False = apenas o primeiro
    SALVAR_TODAS = False      # True = salva tudo / False = salva só variáveis de interesse

    processor = Pl4Processor(
        pasta_entrada="ArquivoPl4",
        pasta_saida="Output",
        modo_todos=PROCESSAR_TODOS,
        salvar_todas=SALVAR_TODAS
    )
    processor.executar()
