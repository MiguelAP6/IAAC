import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def resumo_estatistico_avancado(df):
    """
    Análise Univariada: Calcula Média, Mediana, Desvio Padrão, Skewness e Kurtosis.
    """
    colunas_numericas = df.select_dtypes(include=[np.number]).columns

    estatisticas = []
    for col in colunas_numericas:
        estatisticas.append({
            'Coluna': col,
            'Média': df[col].mean(),
            'Mediana': df[col].median(),
            'Desvio_Padrão': df[col].std(),
            'Assimetria (Skewness)': df[col].skew(),
            'Curtose (Kurtosis)': df[col].kurtosis()
        })

    return pd.DataFrame(estatisticas)

def analise_correlacao(df, metodo='pearson'):
    """
    Calcula e exibe a matriz de correlação (Pearson ou Spearman).
    """
    colunas_numericas = df.select_dtypes(include=[np.number])
    corr = colunas_numericas.corr(method=metodo)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Matriz de Correlação ({metodo.capitalize()})')
    plt.show()

    return corr

def visualizacao_bivariada(df, coluna_num, coluna_alvo='attack_detected'):
    """
    Análise Bivariada: Compara uma variável numérica com a variável alvo de ataques.
    """
    plt.figure(figsize=(12, 5))

    # Histograma/KDE
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=coluna_num, hue=coluna_alvo, kde=True, element="step")
    plt.title(f'Distribuição de {coluna_num} por {coluna_alvo}')

    # Boxplot para identificar Outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x=coluna_alvo, y=coluna_num)
    plt.title(f'Boxplot de {coluna_num} por {coluna_alvo}')

    plt.tight_layout()
    plt.show()

def resumo_dataset(df):
    """Exibe tipos de dados e contagem de nulos."""
    return pd.DataFrame({
        'Tipo_Dado': df.dtypes,
        'Valores_Nulos': df.isnull().sum(),
        'Percentagem_Nulos (%)': (df.isnull().sum() / len(df)) * 100
    })

def distribuicao_target(df, coluna_alvo):
    """Mostra a distribuição das classes da variável alvo."""
    if coluna_alvo in df.columns:
        print(f"\n=== DISTRIBUIÇÃO DA VARIÁVEL ALVO ({coluna_alvo}) ===")
        print(df[coluna_alvo].value_counts())
        print("\nEm percentagem:")
        print(df[coluna_alvo].value_counts(normalize=True) * 100)
