def sigmoid(x):
    e = 2.718
    return 1.0 / (1.0 + (e ** -x))

def relu(x):
    return x if x > 0 else 0.0

def relu_derivada(x):
    return 1.0 if x > 0 else 0.0

def derivada_bce_sigmoid(y_real, y_previsto):
    return y_previsto - y_real

def calcula_perda_absoluta(y_real, y_previsto):
    return abs(y_previsto - y_real)

entradas = [[0, 0], [0, 1], [1, 0], [1, 1]]
saidas_esperadas = [0, 1, 1, 0]

W1 = [
    [ 0.8, -0.5,  0.2], 
    [-0.5,  0.9, -0.3]  
]
b1 = [-0.2, 0.3, 0.1]

W2 = [0.6, -0.4, 0.5] 
b2 = 0.1

taxa_aprendizado = 0.001
epocas = 30000 

print(" INICIANDO TREINAMENTO")
print("====================================================")

for epoca in range(epocas + 1):
    erro_acumulado = 0.0
    
    for i in range(4):
        x1, x2 = entradas[i]
        y_real = saidas_esperadas[i]
        
        z1 = [0.0, 0.0, 0.0]
        a1 = [0.0, 0.0, 0.0] 
        
        for j in range(3):
            soma = (x1 * W1[0][j]) + (x2 * W1[1][j]) + b1[j]
            z1[j] = soma
            a1[j] = relu(soma)
            
        soma_saida = b2
        for j in range(3):
            soma_saida += a1[j] * W2[j]
            
        y_previsto = sigmoid(soma_saida)
        
        #backpropagation
        delta_saida = derivada_bce_sigmoid(y_real, y_previsto)
        
        delta_oculta = [0.0, 0.0, 0.0]
        for j in range(3):
            erro_propagado = delta_saida * W2[j]
            delta_oculta[j] = erro_propagado * relu_derivada(z1[j]) 
            
        #atualização 
        for j in range(3):
            W2[j] -= taxa_aprendizado * delta_saida * a1[j]
        b2 -= taxa_aprendizado * delta_saida
        
        for j in range(3):
            W1[0][j] -= taxa_aprendizado * delta_oculta[j] * x1
            W1[1][j] -= taxa_aprendizado * delta_oculta[j] * x2
            b1[j] -= taxa_aprendizado * delta_oculta[j]
            
        erro_acumulado += calcula_perda_absoluta(y_real, y_previsto)
         
    if epoca % 30 == 0:
        progresso = (epoca / epocas) * 100
        print(f"Progresso: [{progresso:>3.0f}%] | Epoca: {epoca:>5d}/{epocas} | Erro: {erro_acumulado/4:.4f}")

print("\n=========================================================================")
print("                               TESTE FINAL                               ")
print("=========================================================================")
print("| Entrada (x1, x2) | Esperado (y) | Saida (Predicao) | Classe |  Perda  |")
print("|------------------|--------------|------------------|--------|---------|")

for i in range(4):
    x1, x2 = entradas[i]
    y_real = saidas_esperadas[i]
    
    a1 = [0.0, 0.0, 0.0]
    for j in range(3):
        soma = (x1 * W1[0][j]) + (x2 * W1[1][j]) + b1[j]
        a1[j] = relu(soma) 
        
    soma_saida = b2
    for j in range(3):
        soma_saida += a1[j] * W2[j]
        
    y_previsto = sigmoid(soma_saida)
    
    classe = 1 if y_previsto >= 0.5 else 0
    
    perda = calcula_perda_absoluta(y_real, y_previsto) 
    
    print(f"|     [{x1}, {x2}]     |      {y_real}       |      {y_previsto:.4f}      |    {classe}   |  {perda:.4f} |")

print("=========================================================================")