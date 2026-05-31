import numpy as np
from fish import Fish

class School:
    def __init__(self, num_fishes, num_features):
        """
        Inicializa o cardume com uma quantidade específica de peixes.
        """
        self.num_fishes = num_fishes
        self.num_features = num_features
        
        self.fishes = [Fish(num_features) for _ in range(num_fishes)]
        
        # Memória do peso total do cardume
        self.total_weight = float(num_fishes)
        self.prev_total_weight = float(num_fishes)

    def feed(self):
        """
        Distribui o ganho de peso para os peixes que melhoraram seu fitness.
        """
        
        max_delta = max([fish.delta_fitness for fish in self.fishes])
        
        #encerra o processo caso ninguém tenha melhorado
        if max_delta <= 0:
            return

        #Distribui o peso
        for fish in self.fishes:
            if fish.delta_fitness > 0:
                # O ganho de peso é normalizado pelo peixe que teve o maior ganho
                weight_gain = fish.delta_fitness / max_delta
                fish.weight += weight_gain
                
        #Atualiza a balança do cardume
        self.prev_total_weight = self.total_weight
        self.total_weight = sum([fish.weight for fish in self.fishes])

    def instinctive_movement(self, X, y):
        """
        Calcula a direção média de sucesso e puxa o cardume levemente para ela.
        """
        # Soma os ganhos de fitness apenas dos peixes que melhoraram
        sum_delta = sum([f.delta_fitness for f in self.fishes if f.delta_fitness > 0])
        
        # instinto nulo se nenhum peixe melhorou.
        if sum_delta == 0:
            return

        # Calcula o vetor Instinto 
        I = np.zeros(self.num_features)
        for f in self.fishes:
            if f.delta_fitness > 0:
                I += f.position * f.delta_fitness
        I = I / sum_delta

        # Move cada peixe em direção ao Instinto
        for f in self.fishes:
            old_position = np.copy(f.position)
            old_fitness = f.fitness

            for j in range(self.num_features):
                if np.random.rand() < I[j]:
                    f.position[j] = 1
                else:
                    f.position[j] = 0

            # Avalia a nova posição
            f.evaluate(X, y)
            
            # Se o movimento instivido não melhorar o fitness o peixe recusa.
            if f.fitness <= old_fitness:
                f.position = old_position
                f.fitness = old_fitness
                f.delta_fitness = 0.0

    def volitive_movement(self, X, y):
        """
        Expande ou contrai o cardume em relação ao Baricentro.
        """
        # Calcula o Baricentro (B) ponderado pelo peso atual
        B = np.zeros(self.num_features)
        for f in self.fishes:
            B += f.position * f.weight
        B = B / self.total_weight

        # Define a ação volitiva baseada no sucesso global
        is_attracting = self.total_weight > self.prev_total_weight

        # Move o cardume coletivamente
        for f in self.fishes:
            for j in range(self.num_features):
                # Se atrai, a probabilidade de virar '1' é o próprio Baricentro
                # Se repele, a probabilidade inverte (1 - Baricentro) para forçar exploração
                prob_to_one = B[j] if is_attracting else (1.0 - B[j])
                
                if np.random.rand() < prob_to_one:
                    f.position[j] = 1
                else:
                    f.position[j] = 0

            # O peixe aceita o deslocamento mesmo que o fitness caia, para evitar mínimos locais.
            f.evaluate(X, y)