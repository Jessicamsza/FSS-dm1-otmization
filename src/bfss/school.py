import numpy as np
from .fish import Fish

class School:
    def __init__(self, num_fishes, num_features):
        self.num_fishes = num_fishes
        self.num_features = num_features
        self.fishes = [Fish(num_features) for _ in range(num_fishes)]
        
        # Memória de peso global do cardume
        self.total_weight = float(num_fishes)
        self.prev_total_weight = float(num_fishes)

    def feed(self):
        """
        Operador de Alimentação.
        Distribui peso proporcionalmente aos peixes que tiveram sucesso na exploração.
        """
        max_delta = max([fish.delta_fitness for fish in self.fishes])
        
        if max_delta <= 0:
            return

        for fish in self.fishes:
            if fish.delta_fitness > 0:
                # Eq. 3: Variação de peso normalizada pelo maior ganho
                weight_gain = fish.delta_fitness / max_delta
                fish.weight += weight_gain
                
        self.prev_total_weight = self.total_weight
        self.total_weight = sum([fish.weight for fish in self.fishes])

    def instinctive_movement(self, X_train, X_test, y_train, y_test, param_I, weight_acc, weight_feat):
        """
        Movimento Instintivo (Seção 5.b). 
        Atrai o cardume na direção da média ponderada de sucesso individual.
        """
        sum_delta = sum([f.delta_fitness for f in self.fishes if f.delta_fitness > 0])
        if sum_delta == 0:
            return

        # Eq. 13: Cálculo do vetor Instinto Contínuo (I)
        I_continuous = np.zeros(self.num_features)
        for f in self.fishes:
            if f.delta_fitness > 0:
                I_continuous += f.position * f.delta_fitness
        I_continuous = I_continuous / sum_delta

        # Exemplo 15: Binarização via Limiar Adaptativo (Adaptive Threshold)
        threshold = param_I * np.max(I_continuous)
        I_bin = (I_continuous >= threshold).astype(int)

        for f in self.fishes:
            diff_indices = np.where(f.position != I_bin)[0]
            
            # Exemplo 16: Aproximação suave invertendo estritamente 1 único bit diferente
            if len(diff_indices) > 0:
                bit_to_flip = np.random.choice(diff_indices)
                f.position[bit_to_flip] = 1 - f.position[bit_to_flip]

            f.evaluate(X_train, X_test, y_train, y_test, weight_acc, weight_feat)

    def volitive_movement(self, X_train, X_test, y_train, y_test, param_V, weight_acc, weight_feat):
        """
        Movimento Volitivo
        Contrai ou dilata o cardume baseado no sucesso global da iteração.
        """
        # Eq. 6: Cálculo do Baricentro Contínuo ponderado pelos pesos absolutos
        B_continuous = np.zeros(self.num_features)
        for f in self.fishes:
            B_continuous += f.position * f.weight
        B_continuous = B_continuous / self.total_weight

        # Limiar Adaptativo para gerar o Baricentro Binário
        threshold = param_V * np.max(B_continuous)
        B_bin = (B_continuous >= threshold).astype(int)

        # Avalia o sucesso global da iteração
        is_attracting = self.total_weight > self.prev_total_weight

        for f in self.fishes:
            # Exemplo 17 e 18: Se atraindo (sucesso), alvo é o Baricentro. Senão, Anti-Baricentro.
            target = B_bin if is_attracting else (1 - B_bin)
            
            diff_indices = np.where(f.position != target)[0]
            
            # Inversão suave de estritamente 1 único bit
            if len(diff_indices) > 0:
                bit_to_flip = np.random.choice(diff_indices)
                f.position[bit_to_flip] = 1 - f.position[bit_to_flip]

            f.evaluate(X_train, X_test, y_train, y_test, weight_acc, weight_feat)