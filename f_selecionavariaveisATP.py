# filtro_variaveis.py
# -*- coding: utf-8 -*-
"""
Módulo para filtragem de variáveis de arquivos PL4 por padrões de nome.

Uso:
    from filtro_variaveis import seleciona_variaveis
    
    # Com resultado completo do readpl4
    variaveis_filtradas = seleciona_variaveis(
        resultado, 
        filtros=['N', 'GER', 'M_NUM']
    )
    
    # Com lista explícita de nomes
    variaveis_filtradas = seleciona_variaveis(
        resultado,
        nomes_especificos=['SUBA', 'SUBB', 'SUBC']
    )
"""

import re
from typing import Dict, List, Optional, Set
import numpy as np


def seleciona_variaveis(
    resultado: Dict,
    filtros: Optional[List[str]] = None,
    nomes_especificos: Optional[List[str]] = None,
    incluir_time: bool = True
) -> Dict[str, np.ndarray]:
    """
    Seleciona variáveis de interesse a partir do resultado do readpl4.
    
    Parâmetros:
    -----------
    resultado : dict
        Dicionário retornado pela função readpl4, contendo:
        - 'time': array de tempo
        - 'data': matriz (n_var, n_samples)
        - 'labels': lista de nomes das variáveis
        
    filtros : list de str, opcional
        Lista de filtros para selecionar variáveis por padrão. Opções:
        - 'N': variáveis que começam com 'N' seguido de dígito (ex: N02FA, N03FB, N09F)
        - 'GER': variáveis que começam com 'GER' (ex: GERA, GERB, GERC)
        - 'M_NUM': variáveis que começam com 'M' seguido de número (ex: M02FA, M11FB, M30F)
        - 'GS': variáveis que começam com 'GS' (ex: GS1A, GS1B, GS1C)
        - 'SUB': variáveis que começam com 'SUB' (ex: SUBA, SUBB, SUBC, SUBNEU)
        - 'time': inclui a variável de tempo (redundante se incluir_time=True)
        
    nomes_especificos : list de str, opcional
        Lista explícita de nomes de variáveis a serem selecionadas.
        Se fornecido, tem prioridade sobre 'filtros'.
        
    incluir_time : bool, default=True
        Se True, sempre inclui a variável 'time' no resultado.
        
    Retorna:
    --------
    dict
        Dicionário com estrutura {'time': array, 'Var1': array, 'Var2': array, ...}
        Pronto para ser usado com save_results_parquet(usar_variaveis_interesse=True)
        
    Exemplos:
    ---------
    # Selecionar apenas variáveis que começam com 'N' e 'GER'
    >>> medicoes = seleciona_variaveis(resultado, filtros=['N', 'GER'])
    
    # Selecionar variáveis específicas
    >>> medicoes = seleciona_variaveis(
    ...     resultado, 
    ...     nomes_especificos=['SUBA', 'SUBB', 'SUBC']
    ... )
    
    # Combinar vários filtros
    >>> medicoes = seleciona_variaveis(
    ...     resultado,
    ...     filtros=['N', 'GER', 'M_NUM', 'GS', 'SUB']
    ... )
    """
    
    # Validação de entrada
    if "time" not in resultado or "data" not in resultado or "labels" not in resultado:
        raise ValueError(
            "O dicionário 'resultado' deve conter as chaves 'time', 'data' e 'labels'"
        )
    
    time = np.asarray(resultado["time"])
    data = np.asarray(resultado["data"])
    labels = list(resultado["labels"])
    
    if data.shape[0] != len(labels):
        raise ValueError(
            f"Incompatibilidade: data tem {data.shape[0]} variáveis, "
            f"mas labels tem {len(labels)} elementos"
        )
    
    # Criar dicionário de saída
    out: Dict[str, np.ndarray] = {}
    
    # Sempre incluir time se solicitado
    if incluir_time:
        out["time"] = time
    
    # Determinar quais variáveis selecionar
    indices_selecionados: Set[int] = set()
    
    # Opção 1: Lista explícita de nomes (prioridade)
    if nomes_especificos is not None:
        nomes_set = set(nomes_especificos)
        for i, label in enumerate(labels):
            # Extrai o nome base da variável (remove sufixos de tipo entre parênteses)
            nome_base = _extrair_nome_base(label)
            if nome_base in nomes_set or label in nomes_set:
                indices_selecionados.add(i)
        
        print(f"  🔍 Seleção por nomes específicos: {len(indices_selecionados)} variáveis encontradas")
    
    # Opção 2: Filtros por padrão
    elif filtros is not None:
        for filtro in filtros:
            filtro_upper = str(filtro).upper().strip()
            
            if filtro_upper == 'N':
                # Variáveis que começam com 'N' seguido de dígito
                pattern = re.compile(r'^N\d', re.IGNORECASE)
                for i, label in enumerate(labels):
                    nome_base = _extrair_nome_base(label)
                    if pattern.match(nome_base):
                        indices_selecionados.add(i)
            
            elif filtro_upper == 'GER':
                # Variáveis que começam com 'GER'
                for i, label in enumerate(labels):
                    nome_base = _extrair_nome_base(label)
                    if nome_base.upper().startswith('GER'):
                        indices_selecionados.add(i)
            
            elif filtro_upper == 'M_NUM' or filtro_upper == 'M':
                # Variáveis que começam com 'M' seguido de número
                pattern = re.compile(r'^M\d', re.IGNORECASE)
                for i, label in enumerate(labels):
                    nome_base = _extrair_nome_base(label)
                    if pattern.match(nome_base):
                        indices_selecionados.add(i)
            
            elif filtro_upper == 'GS':
                # Variáveis que começam com 'GS'
                for i, label in enumerate(labels):
                    nome_base = _extrair_nome_base(label)
                    if nome_base.upper().startswith('GS'):
                        indices_selecionados.add(i)
            
            elif filtro_upper == 'SUB':
                # Variáveis que começam com 'SUB'
                for i, label in enumerate(labels):
                    nome_base = _extrair_nome_base(label)
                    if nome_base.upper().startswith('SUB'):
                        indices_selecionados.add(i)
            
            elif filtro_upper == 'TIME':
                # Redundante se incluir_time=True, mas aceito
                pass
            
            else:
                print(f"  ⚠️  Filtro desconhecido: '{filtro}' (ignorado)")
        
        print(f"  🔍 Seleção por filtros {filtros}: {len(indices_selecionados)} variáveis encontradas")
    
    # Se nenhum critério foi fornecido, retorna todas
    else:
        indices_selecionados = set(range(len(labels)))
        print(f"  🔍 Nenhum filtro especificado: retornando todas as {len(labels)} variáveis")
    
    # Adicionar variáveis selecionadas ao dicionário
    for i in sorted(indices_selecionados):
        nome_var = labels[i]
        out[nome_var] = data[i, :]
    
    return out


def _extrair_nome_base(label: str) -> str:
    """
    Extrai o nome base da variável, removendo informações de tipo.
    
    Exemplos:
        'SUBA- (V-node)' -> 'SUBA'
        'N02FA-M02FA (I-bran)' -> 'N02FA-M02FA'
        'GERA-SUBA (I-bran)' -> 'GERA-SUBA'
    """
    # Remove parte entre parênteses (tipo de sinal)
    if '(' in label:
        label = label.split('(')[0].strip()
    
    # Remove traço final isolado se existir
    if label.endswith('-'):
        label = label[:-1].strip()
    
    # Para labels com hífen (ex: 'N02FA-M02FA'), usa a primeira parte
    if '-' in label:
        # Mas apenas se não for parte do nome (ex: 'GERA-SUBA' mantém)
        partes = label.split('-')
        if len(partes) == 2:
            # Retorna a primeira parte (a mais relevante para filtragem)
            return partes[0].strip()
    
    return label.strip()


def listar_variaveis_por_filtro(
    resultado: Dict,
    filtro: str,
    mostrar: bool = True
) -> List[str]:
    """
    Lista todas as variáveis que correspondem a um filtro específico.
    Útil para verificar quais variáveis serão selecionadas.
    
    Parâmetros:
    -----------
    resultado : dict
        Dicionário retornado pela função readpl4
    filtro : str
        Um dos filtros: 'N', 'GER', 'M_NUM', 'GS', 'SUB'
    mostrar : bool, default=True
        Se True, imprime a lista na tela
        
    Retorna:
    --------
    list de str
        Lista de nomes de variáveis que correspondem ao filtro
    """
    labels = list(resultado.get("labels", []))
    variaveis_encontradas = []
    
    filtro_upper = str(filtro).upper().strip()
    
    if filtro_upper == 'N':
        pattern = re.compile(r'^N\d', re.IGNORECASE)
        variaveis_encontradas = [
            label for label in labels 
            if pattern.match(_extrair_nome_base(label))
        ]
    elif filtro_upper == 'GER':
        variaveis_encontradas = [
            label for label in labels 
            if _extrair_nome_base(label).upper().startswith('GER')
        ]
    elif filtro_upper in ['M_NUM', 'M']:
        pattern = re.compile(r'^M\d', re.IGNORECASE)
        variaveis_encontradas = [
            label for label in labels 
            if pattern.match(_extrair_nome_base(label))
        ]
    elif filtro_upper == 'GS':
        variaveis_encontradas = [
            label for label in labels 
            if _extrair_nome_base(label).upper().startswith('GS')
        ]
    elif filtro_upper == 'SUB':
        variaveis_encontradas = [
            label for label in labels 
            if _extrair_nome_base(label).upper().startswith('SUB')
        ]
    
    if mostrar:
        print(f"\n📋 Variáveis com filtro '{filtro}': {len(variaveis_encontradas)} encontradas")
        for i, var in enumerate(variaveis_encontradas, 1):
            print(f"  {i:3d}. {var}")
    
    return variaveis_encontradas


def listar_todas_variaveis(resultado: Dict, mostrar: bool = True) -> List[str]:
    """
    Lista todas as variáveis disponíveis no resultado.
    
    Parâmetros:
    -----------
    resultado : dict
        Dicionário retornado pela função readpl4
    mostrar : bool, default=True
        Se True, imprime a lista na tela
        
    Retorna:
    --------
    list de str
        Lista completa de nomes de variáveis
    """
    labels = list(resultado.get("labels", []))
    
    if mostrar:
        print(f"\n📋 Total de variáveis disponíveis: {len(labels)}")
        for i, label in enumerate(labels):
            print(f"  {i:3d}. {label}")
    
    return labels


# Exemplo de uso
if __name__ == "__main__":
    print("=" * 70)
    print("MÓDULO: filtro_variaveis.py")
    print("=" * 70)
    print("\nEste módulo fornece funções para filtrar variáveis de arquivos PL4.")
    print("\nFiltros disponíveis:")
    print("  • 'N'     : Variáveis que começam com 'N' + número (ex: N02FA, N03FB)")
    print("  • 'GER'   : Variáveis que começam com 'GER' (ex: GERA, GERB, GERC)")
    print("  • 'M_NUM' : Variáveis que começam com 'M' + número (ex: M02FA, M11FB)")
    print("  • 'GS'    : Variáveis que começam com 'GS' (ex: GS1A, GS1B)")
    print("  • 'SUB'   : Variáveis que começam com 'SUB' (ex: SUBA, SUBB)")
    print("\nUso básico:")
    print("  from filtro_variaveis import seleciona_variaveis")
    print("  medicoes = seleciona_variaveis(resultado, filtros=['N', 'GER'])")
    print("=" * 70)