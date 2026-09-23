import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def treinar_e_avaliar_modelos(X, y):
    """
    Divide os dados, treina três modelos clássicos e devolve o melhor modelo.
    """
    # 1. Dividir em dados de treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Definir os modelos
    modelos = {
        'Regressao_Logistica': LogisticRegression(max_iter=1000),
        'Arvore_Decisao': DecisionTreeClassifier(random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    melhor_modelo = None
    melhor_score = 0
    nome_melhor_modelo = ""

    print("=== TREINO E AVALIAÇÃO DE MODELOS ===")
    # 3. Treinar e testar cada modelo
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        previsoes = modelo.predict(X_test)

        precisao = accuracy_score(y_test, previsoes)
        print(f"\nModelo: {nome}")
        print(f"Precisão Global (Accuracy): {precisao:.4f}")

        # Guardar o melhor modelo
        if precisao > melhor_score:
            melhor_score = precisao
            melhor_modelo = modelo
            nome_melhor_modelo = nome

    print(f"\n🏆 O MELHOR MODELO FOI: {nome_melhor_modelo} (Precisão: {melhor_score:.4f})")

    return melhor_modelo, nome_melhor_modelo

def guardar_modelo(modelo, nome_modelo, caminho_pasta):
    """
    Guarda o modelo treinado na pasta especificada.
    """
    caminho_completo = f"{caminho_pasta}/{nome_modelo}.pkl"
    joblib.dump(modelo, caminho_completo)
    print(f"\nModelo guardado com sucesso em: {caminho_completo}")
