import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

def tratar_outliers_iqr(df, colunas):
    """Aplica clipping para limitar Outliers usando o método IQR."""
    df_out = df.copy()
    for col in colunas:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df_out[col] = np.clip(df_out[col], limite_inferior, limite_superior)
    return df_out

def engenharia_de_features(df):
    """Cria novas variáveis (Feature Engineering/Extraction)."""
    df_eng = df.copy()
    # Criar um rácio de falhas de login (evitando divisão por zero)
    df_eng['fail_login_ratio'] = df_eng['failed_logins'] / (df_eng['login_attempts'] + 1)
    return df_eng

def preparar_dados_completo(df, coluna_alvo='attack_detected'):
    """Executa toda a pipeline do Módulo 4 de Preparação de Dados."""
    df_clean = df.copy()
    
    # 1. Data Cleaning (Missing Values & Remoção de IDs)
    if 'session_id' in df_clean.columns:
        df_clean = df_clean.drop(columns=['session_id'])
    if 'encryption_used' in df_clean.columns:
        df_clean['encryption_used'] = df_clean['encryption_used'].fillna('Desconhecido')
        
    # 2. Feature Engineering
    df_clean = engenharia_de_features(df_clean)
    
    # 3. Tratamento de Outliers (Apenas em features contínuas)
    cols_continuas = ['network_packet_size', 'session_duration', 'ip_reputation_score']
    df_clean = tratar_outliers_iqr(df_clean, cols_continuas)
    
    # 4. Feature Extraction (Categoricals para Numéricas)
    colunas_texto = df_clean.select_dtypes(include=['object']).columns.tolist()
    df_clean = pd.get_dummies(df_clean, columns=colunas_texto, drop_first=True)
    
    X = df_clean.drop(columns=[coluna_alvo])
    y = df_clean[coluna_alvo]
    
    # 5. Splitting Data (Divisão de Dados com estratificação)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Feature Scaling (Normalização)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # 7. Feature Selection (Selecionar as melhores variáveis)
    selector = SelectKBest(score_func=f_classif, k=min(10, X_train_scaled.shape[1]))
    selector.fit(X_train_scaled, y_train)
    cols_selecionadas = X_train_scaled.columns[selector.get_support()]
    
    X_train_final = X_train_scaled[cols_selecionadas]
    X_test_final = X_test_scaled[cols_selecionadas]
    
    # 8. Imbalanced Data (Equilibrar com SMOTE no treino)
    # Embora os nossos dados estejam relativamente equilibrados, aplicamos para cumprir currículo
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train)
    
    print(f"Preparação concluída! Variáveis selecionadas: {list(cols_selecionadas)}")
    return X_train_resampled, X_test_final, y_train_resampled, y_test, scaler
