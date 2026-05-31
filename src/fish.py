import numpy as np

class Fish:
    def __init__(self, num_features):
        """
        Inicializa um peixe para o Binary Fish School Search.
        :param num_features: total de atributos no dataset.
        """
        self.position = np.random.randint(2, size=num_features)  # Posição binária
        self.weight = 1.0  # Peso inicial
        self.fitness = 0.0  
        self.delta_fitness = 0.0  # Variação do fitness
        
    def evaluate(self, X_train, y_train):
        """
        Recebe o dataset e treina um modelo apenas com os atributos '1' em self.position.
        """
        print(f"O peixe está na posição: {self.position}")
        # Aqui entrará a lógica do modelo de Machine Learning
        pass
    