import os
import pandas as pd
from bfss.school import School

def load_data(filepath, target_column='VALOR_RESULTADO'):
    """
    Carrega a base de dados.
    Default Target = hemoglobina glicada.
    """
    df = pd.read_csv(filepath)
    
    # Separa a variável alvo dos atributos preditivos
    y = df[target_column].values
    X_df = df.drop(columns=[target_column])
    
    feature_names = X_df.columns.tolist()
    X = X_df.values
    
    return X, y, feature_names

def main():
    #ALTERAR AQUI PARA O CAMINHO DO DATASET
    filepath = "data/dataset.csv"  
    

    if not os.path.exists(filepath):
        print(f"Arquivo de dados não encontrado em: {filepath}")
        return

    print("Carregando banco de dados...")
    X, y, feature_names = load_data(filepath)
    
    # Configurações do algoritmo
    num_fishes = 30          
    num_features = X.shape[1] 
    num_iterations = 50      
    
    print(f"Criando o Cardume com {num_fishes} peixes...")
    cardume = School(num_fishes=num_fishes, num_features=num_features)
    
    print("Iniciando a Seleção de atributos...")
    for i in range(num_iterations):      
        # ETAPA 1: Movimento Individual
        for fish in cardume.fishes:
            fish.individual_movement(X, y)
            
        # ETAPA 2: Alimentação
        cardume.feed()
        
        # ETAPA 3: Movimento Instintivo Coletivo
        cardume.instinctive_movement()
        
        # ETAPA 4: Movimento Volitivo Coletivo
        cardume.volitive_movement()
        
    print("Fim das iterações.")
    
    melhor_peixe = max(cardume.fishes, key=lambda f: f.fitness)
    print(f"Acurácia Máxima Alcançada: {melhor_peixe.fitness:.4f}")
    
    colunas_escolhidas = [feature_names[i] for i in range(num_features) if melhor_peixe.position[i] == 1]
    
    print(f"\nO bFSS selecionou os seguintes atributos para a próxima etapa (MOFSS):")
    for col in colunas_escolhidas:
        print(f" - {col}")

if __name__ == "__main__":
    main()