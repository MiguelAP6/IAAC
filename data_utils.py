import pandas as pd

def carregar_dados(caminho_ficheiro):
    """
    Função auxiliar para carregar o dataset de cibersegurança e validar o formato.
    """
    try:
        df = pd.read_csv(caminho_ficheiro)
        print(f"Sucesso! O dataset tem {df.shape[0]} linhas e {df.shape[1]} colunas.")
        return df
    except Exception as e:
        print(f"Erro ao carregar o ficheiro: {e}")
        return None
