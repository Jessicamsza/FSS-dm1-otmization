import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from .school import School

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_data(filepath, target_column):
    """
    Carrega a base de dados e separa alvo de preditores.
    """
    df = pd.read_csv(filepath)
    y = df[target_column].values
    X_df = df.drop(columns=[target_column])
    return X_df.values, y, X_df.columns.tolist()

def run_bfss(filepath, num_fishes=30, num_iterations=50, target_column='VALOR_RESULTADO'):
    """
    Executa o algoritmo BFSS para seleção de atributos.
    target padrão: 'VALOR_RESULTADO' (hemoglobina glicada).
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    logging.info("Carregando dados...")
    X, y, feature_names = load_data(filepath, target_column)
    
    # Split fixo para garantir a estabilidade do Wrapper KNN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  
    num_features = X.shape[1] 
    
    # Hiperparâmetros recomendados pelo artigo base
    s_ind_start = 0.1       # Passo individual inicial
    s_ind_end = 0.001       # Passo individual final
    param_I = 0.4           # Modificador do Threshold Instintivo
    param_V = 0.4           # Modificador do Threshold Volitivo
    weight_acc = 0.99       # Peso (Alpha) para performance na Eq. 10
    weight_feat = 0.01      # Peso (Beta) para redução dimensional na Eq. 10
    
    logging.info(f"Criando Cardume: {num_fishes} peixes e {num_features} atributos.")
    cardume = School(num_fishes, num_features)
    
    logging.info("Iniciando a Seleção de Atributos (BFSS)...")
    for t in range(num_iterations):      
        
        # Eq. 4: Atualização do decaimento linear de S_ind(t)
        s_ind_t = s_ind_start - ((s_ind_start - s_ind_end) / num_iterations) * t

        # Etapa 1: Exploração Individual
        for fish in cardume.fishes:
            fish.individual_movement(X_train, X_test, y_train, y_test, s_ind_t, weight_acc, weight_feat)
            
        # Etapa 2: Alimentação e Atualização de Pesos
        cardume.feed()
        
        # Etapa 3 e 4: Movimentação Coletiva (Instinto e Volição)
        cardume.instinctive_movement(X_train, X_test, y_train, y_test, param_I, weight_acc, weight_feat)
        cardume.volitive_movement(X_train, X_test, y_train, y_test, param_V, weight_acc, weight_feat)
        
        # Monitoramento
        melhor_iteracao = max(cardume.fishes, key=lambda f: f.fitness)
        qtd_features_ativas = sum(melhor_iteracao.position)
        logging.debug(f"Iteração {t+1:02d}/{num_iterations} | Fitness: {melhor_iteracao.fitness:.4f} | Atributos: {qtd_features_ativas:02d}")

    logging.info("\nFim do Treinamento.")
    
    # Extração de Conhecimento
    melhor_peixe_global = max(cardume.fishes, key=lambda f: f.fitness)
    selected_features= [feature_names[i] for i in range(num_features) if melhor_peixe_global.position[i] == 1]
    
    logging.info(f"O algoritmo selecionou {len(selected_features)} dos {num_features} atributos iniciais.")
    for col in selected_features:
        print(f" {col}")
    
    return selected_features