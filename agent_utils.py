import joblib
from transformers import pipeline

def inicializar_agente_llm():
    """
    Carrega um modelo LLM leve e rápido da Hugging Face.
    """
    print("A carregar o Agente LLM (pode demorar ~1 minuto na GPU)...")
    gerador = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")
    return gerador

def analisar_e_relatar(caminho_modelo, amostra_dados, agente_llm):
    """
    Consulta o classificador Random Forest e pede ao LLM para redigir o alerta.
    """
    # 1. Carregar o modelo treinado
    classificador = joblib.load(caminho_modelo)

    # 2. Prever a classe
    previsao = classificador.predict(amostra_dados)[0]
    estado = "Malicioso (Possível Intrusão)" if previsao == 1 else "Benigno (Tráfego Normal)"

    # 3. Prompt de Cibersegurança
    prompt = f"<|system|>\nYou are an expert cybersecurity analyst.\n<|user|>\nA machine learning model detected a network packet with status: {estado}. Write a short security alert report (2 concise sentences) for the SOC team.\n<|assistant|>\n"

    # 4. Gerar resposta
    print(f"\n[Modelo Tradicional ML] Diagnóstico: {estado}")
    print("[Agente LLM Generativo] A redigir relatório de incidente...")

    resposta = agente_llm(prompt, max_new_tokens=100, return_full_text=False, temperature=0.7)
    return resposta[0]['generated_text'].strip()
