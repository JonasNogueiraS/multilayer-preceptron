# MultiLayer Perceptron (MLP) do Zero: Problema do XOR

Uma implementação didática de uma rede neural artificial multicamadas construída inteiramente em **Python puro**, sem a utilização de bibliotecas externas (como NumPy, Pandas ou Scikit-Learn). O projeto demonstra o funcionamento do algoritmo de **Backpropagation** para resolver o clássico problema lógico do XOR (OU Exclusivo).

## 🧠 Sobre o Projeto

O problema do XOR é historicamente significativo na Inteligência Artificial por não ser linearmente separável, o que exige a utilização de pelo menos uma camada oculta para sua resolução. Esta implementação foca na "matemática raiz", processando cada conexão e ajuste de peso escalar por escalar.

### Especificações Técnicas
* **Arquitetura**: 2 neurônios de entrada, 3 neurônios na camada oculta e 1 neurônio de saída.
* **Camada Oculta**: Utiliza a função de ativação **ReLU** ($max(0, x)$).
* **Camada de Saída**: Utiliza a função **Sigmoid** para fornecer uma saída probabilística entre 0 e 1.
* **Função de Custo**: Baseada em **Binary Cross-Entropy (BCE)**.
* **Otimizador**: Descida do Gradiente Estocástica (SGD) com taxa de aprendizado fixa.

## 🛠️ Estrutura do Código

O script foi organizado para separar as responsabilidades matemáticas, facilitando o estudo de cada etapa:

1.  **Funções de Ativação**: Implementação manual da Sigmoid e ReLU, além de suas respectivas derivadas.
2.  **Cálculo de Erro**: Função dedicada para a derivada da BCE combinada com a Sigmoid ($y_{previsto} - y_{real}$).
3.  **Feedforward**: Processamento das entradas através de somas ponderadas e ativações.
4.  **Backpropagation**: Retropropagação do erro da saída para a camada oculta.
5.  **Atualização**: Ajuste dos pesos ($W$) e bias ($b$) com base nos gradientes calculados.

## 🚀 Como Executar

Não há necessidade de instalar dependências. Basta ter o Python instalado.

1. Clone o repositório:
   ```bash
   git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
