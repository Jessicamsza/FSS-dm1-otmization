import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Fish:
    def __init__(self, num_features):
        """
        Inicializa um peixe para o Binary Fish School Search.
        :param num_features: total de atributos no dataset.
        """
        # Eq. 11: Posição inicial gerada aleatoriamente com distribuição uniforme (binário)
        self.position = np.random.randint(2, size=num_features) 
        self.weight = 1.0 
        self.fitness = 0.0  
        self.delta_fitness = 0.0

        self.num_features = num_features
        
    def evaluate(self, X_train, X_test, y_train, y_test, weight_acc, weight_feat):
        """
        Abordagem Wrapper com KNN para avaliar o subconjunto atual de atributos.
        """
        selected_features = np.where(self.position == 1)[0]
        num_selected = len(selected_features)

        # Penalidade máxima se o peixe desativar todos os atributos
        if num_selected == 0:
            self.delta_fitness = 0.0 - self.fitness
            self.fitness = 0.0 
            return
        
        # Treina e testa o KNN apenas com os atributos ativos (1)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train[:, selected_features], y_train)
        predictions = model.predict(X_test[:, selected_features])
        
        acc = accuracy_score(y_test, predictions)

        # Eq. 10: Fitness pondera a Performance (Acurácia) e a Redução de Dimensionalidade
        feature_penalty = (self.num_features - num_selected) / self.num_features
        current_fitness = (weight_acc * acc) + (weight_feat * feature_penalty)
        
        # Atualiza a memória de variação (Delta) para a etapa de alimentação
        self.delta_fitness = current_fitness - self.fitness
        self.fitness = current_fitness

    def individual_movement(self, X_train, X_test, y_train, y_test, s_ind_t, weight_acc, weight_feat):
        """
        Exploração local baseada na probabilidade de decaimento S_ind.
        """
        old_position = np.copy(self.position)
        old_fitness = self.fitness
        
        # Eq. 12: Inversão condicional (flip) para cada bit do vetor
        for j in range(self.num_features):
            if np.random.rand() < s_ind_t:
                self.position[j] = 1 - self.position[j] 
                
        self.evaluate(X_train, X_test, y_train, y_test, weight_acc, weight_feat)
        
        # Condição de Aceitação: o peixe só adota a nova posição se o fitness melhorar
        if self.fitness <= old_fitness:
            self.position = old_position
            self.fitness = old_fitness
            self.delta_fitness = 0.0