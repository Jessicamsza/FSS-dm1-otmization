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
        self.position = np.random.randint(2, size=num_features)  # Posição binária inicial aleatoriezada
        self.weight = 1.0  # Peso inicial
        self.fitness = 0.0  
        self.delta_fitness = 0.0  # Variação do fitness
        
    def evaluate(self, X, Y):
        """
        Recebe o dataset e treina um modelo apenas com os atributos '1' em self.position.
        Usa KNN como avaliador na abordagem Wapper.
        """
        print(f"O peixe está na posição: {self.position}")
        
        selected_features = np.where(self.position == 1)[0]  # Índices dos atributos selecionado

        if len(selected_features) == 0:
            self.fitness = 0.0  # Se nenhum atributo for selecionado, fitness é zero
            return
        
        X_subset = X[:, selected_features] # Subconjunto com os atributos selecionados

        #dividir o dataset em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X_subset, Y, test_size=0.3, random_state=42)

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        #Calcula o Fitness (Acurácia)
        current_fitness = accuracy_score(y_test, predictions)
        
        #Atualiza a memória interna do peixe
        self.delta_fitness = current_fitness - self.fitness
        self.fitness = current_fitness

        def individual_movement(self, X, y):
            """
            Realiza a exploração local: inverte alguns bits de sua posição e avalia a combinação.
            """
            # Salva o estado atual na memória
            old_position = np.copy(self.position)
            old_fitness = self.fitness
            
            # Escolhe um atributo aleatório da posição para inverter
            feature_to_flip = np.random.randint(len(self.position))
            self.position[feature_to_flip] = 1 - self.position[feature_to_flip]
            
            # Avalia a nova posição
            self.evaluate(X, y)
            
            # só aceita a mudança se o fitness melhorar, caso contrário, desfaz a inversão.
            if self.fitness <= old_fitness:
                self.position = old_position
                self.fitness = old_fitness
                self.delta_fitness = 0.0 