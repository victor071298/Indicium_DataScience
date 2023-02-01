import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn import metrics
from IPython.display import display
import plotly.express as px
from collections import Counter
from dataprep.eda import create_report
from dataprep.datasets import load_dataset

#Função responsável por receber o caminho dos arquivos CSV desejados e retorná-los 
def busca_arquivos(caminho_csv):
    return pd.read_csv(caminho_csv)

#Função responsável por, dada uma entrada de database com colunas treino e colunas treinadas gera nossa árvore de decisão
def gera_arvore(arvore,treino, treinada):
    return arvore.fit(treino,treinada)



def main():

    ##Buscando os Dados
    
    #Deve ser informado o Path em que os arquivos se encontram (lembrando que é necessário usar barras duplas no lugar de barras para evitar erros de leitura do arquivo)
    conjunto_treino = busca_arquivos('Dados\\desafio_manutencao_preditiva_treino.csv')
    conjunto_teste = busca_arquivos('Dados\\desafio_manutencao_preditiva_teste.csv')
    
    #Para visualizar o estado dos conjuntos de treino e teste, basta remover o comentário das duas linhas abaixo
    #display(conjunto_treino)
    #display(conjunto_teste)
    
    
    ##Tratamento de Dados
    
    #Utilizando a bilioteca dataprep do Python, irei gerar um relatório de EDA para nossa base de dados
    #O report gerado será anexado em pdf, mas caso queira gerar ele novamente, basta tirar o comentário da linha abaixo
    #create_report(conjunto_treino, title = 'Relatorio  de EDA').show_browser()
    
    #Mais abaixo, será feito também um gráfico para melhor visualização dos dados com relação ao tipo de falha
    
    #No relatório gerado, podemos ver que não há linhas ou colunas vazias, portanto não precisamos nos preocupar com isso em nossa base de dados
    #Além disso, como 'udi' e 'product id' sao apenas indentificadores, eles serão removidos por não serem necessários no gráfico ou no processo de Machine Learning
    #Essa informação também pode ser observada no relatorio de EDA em Dataset Insights
    conjunto_treino.drop('udi',inplace=True,axis=1)
    conjunto_teste.drop('udi',inplace=True,axis=1)
    conjunto_treino.drop('product_id',inplace=True,axis=1)
    conjunto_teste.drop('product_id',inplace=True,axis=1)
    
    #Para visualizar o novo estado dos conjuntos, basta remover o comentário das duas linhas abaixo
    #display(conjunto_treino)
    #display(conjunto_teste)
    
    
    ##Análise de Dados
    
    #Para visualizar a proporção de falhas do conjunto de treino, basta remover o comentário das três linhas abaixo
    #display(conjunto_treino['failure_type'].value_counts())
    #print()
    #display(conjunto_treino['failure_type'].value_counts(normalize=True).map('{:.1%}'.format))
    
    '''
        Como random failures ocorrem independentemente dos parâmetros, eles serão desconsiderados uma vez que durante o processo de aprendizagem de máquina dados imprecisos
        sejam gerados, atrapalhando o processo de Machine Learning. Além disso, podemos ver que 0.2%  de nossa database possui random failures, então ainda 
        que removamos esse dado ainda teremos mais de 99%  de nossa base de dados para trabalharmos.
    '''
    
    #Removendo 'Random Failure' de nossa base de dados
    conjunto_treino_sem_random = conjunto_treino.copy()
    conjunto_treino_sem_random = conjunto_treino_sem_random[conjunto_treino_sem_random.failure_type != 'Random Failures']
    
    #Para visualizar o estado do conjunto de treino sem random failures basta tirar o comentário das duas linhas abaixo
    #display(conjunto_treino_sem_random)
    #display(conjunto_treino_sem_random['failure_type'].value_counts(normalize=True).map('{:.1%}'.format))
    
    
    ##Análise Gráfica
    
    #Agora faremos gráficos para tentar entender melhor como os erros se comportam em relação as caracteristicas das máquinas
    #Como queremos identificar as principais estatísticas descritivas, iremos gerar um gráfico com informações referentes a cada dado relevante da coluna
    #A coluna 'failure_type' será mostrada como parâmetro 'cor' nos gráficos abaixo para vermos sua modificação para diferentes valores da coluna
    
    #Gerando os gráficos
    for coluna in conjunto_treino_sem_random:
        if coluna!= 'failure_type':
            fig = px.histogram(conjunto_treino_sem_random,x=coluna,color='failure_type')
            fig.show()
 

    #Podemos alterar o gráfico, de maneira que visualizemos apenas uma informação, ou mais de uma ao mesmo tempo
    #Para poder analisar de maneira mais precisa, visualizei cada informação indivudual, comparando-a com o valor total para ter uma ideia dos padrôes proporcionalmente
    #As conclusões obtidas referentes a cada gráfico foram as seguintes:
    
    
    ##Interpretação Gráfica
    
    ''' 
        Quanto ao gráfico sobre o tipo de máquina, podemos ver que quase todos os gráficos mantêm um padrao aproximado relativo as suas quantidades, com exceção de um, o erro 
        de esforço excessivo (overstrain failure). Podemos ver que falhas desse tipo são mais comuns nas máquinas do tipo L, ou pelo menos, mais raras nas de tipo M.
        
        Quanto ao gráfico sobre temperatura do ar, vemos que a maioria dos dados estão bem esparsos e sem padrões aparentes, entretanto, podemos observar que erros de dissipação 
        de calor (heat dissipation failure) se concentram em uma faixa específica, entre 300.9k e 303.7k.
        
        Quanto ao gráfico sobre temperatura de processo, vemos que o caso anterior se repete, a maioria dos erros estão esparsos com exceção do erro de dissipação de calor citado antereior-
        mente, que se concentra entre 309.4k e 312.2k.
        
        Quanto ao gráfico de velocidade rotacional, podemos notar que os casos de erro por falta de energia (power failure) não se encontram entre 1480k e 2259k, enquanto os erros de
        esforço excessivo e dissipação de calor so ocorrem na faixa de 1180k ate 1519k e 1240k ate 1379k respectivamente.
        
        Quanto ao gráfico de torque podemos remover informação sobre todos os tipos de falha, sendo eles que entre 15 e 59 de torque não ocorrem falhas de energia, e as falhas de uso
        excessivo, dissipação de calor e uso de ferramenta, ocorrem nas faixas de 46 a 68.9 , 41 a 67.9 e 16 a 47.9 respectivamente, valores bem distintos para cada erro.
        
        Por fim, no gráfico de gasto de ferramenta por minuto, podemos ver que os erros de gasto de ferramenta e esforço excessivo ocorrem em uma faixa específica, sendo ela de
        200 a 239 por minuto e 180 a 254 por minuto respectivamente.
    '''
    
    
    ##Prevendo os Tipos de Falhas
    
    '''
        Como podemos ver na interpretação dos gráficos gerados, após uma análise cuidadosa percebemos que alguns padrões ocorrem apenas em casos mais específicos, ou são mais 
        facilmente determinados em certas situações. Devido a isso, podemos tentar gerar uma árvore de decisão utilizando esses dados para medir o comportamento esperado.
    '''
    
    
    ##Tipo de Problema
    
    '''
        Como queremos determinar um tipo de erro específico, o problema que estamos resolvendo é um problema de classificação. Isso ocorre pois temos diferentes tipos de falhas,
        e queremos determinar dentre um conjunto possível qual a falha correta (incluindo a falta dela) para aquela máquina.
    '''
    
        
    ##Método Escolhido
    
    '''
        Como a árvore de decisão é um método versátil, não linear e que nos permite ver o impacto de diferentes decisões adotadas, ela será utilizada neste problema.
        Esse método irá estimar os resultados do nosso conjunto de teste utilizando o conjunto de treino, mas, para testar sua eficácia e analisarmos os melhores dados na 
        geração da árvore, será necessário subdividir nosso conjunto de treino, uma vez que com o conjunto de teste que nos foi fornecido não podemos fazer testes de acurácia
        para estimarmos a eficiência da árvore.
        Devido a isso, 2/3 do nosso conjunto de treino se manterá como de treino e gerará uma árvore exemplo, enquanto 1/3 será utilizado como conjunto de teste para podermos 
        ver como nossa árvore está se comportando e que escolhas geram um impacto melhor ou pior. Essa proporção foi escolhida pois é a mesma dos nossos dados inicias, onde 
        temos de 1 até 6666 (2/3 de 10000) no conjunto de treino e de 6667 até 10000 (1/3 de 10000) no de teste.
        Com isso poderemos estimar quais as melhores escolhas durante a criação da árvore, além de ver se elas foram coerentes e, mais importante, observar quais outras escolhas
        podem impactar positivamente na nossa árvore de decisão.
    '''
    
    #Subdividindo o conjunto de treino em cerca de 2/3 e 1/3, como novos conjuntos temporários de treino e teste, temos:
    conjunto_treino_1 = conjunto_treino_sem_random.iloc[0:4444,:]
    
    #Como o teste 2 é usado para estimar o valor real, utilizaremos a base de dados original com as random failures para ver a precisão real que temos
    conjunto_teste_2 = conjunto_treino.iloc[4445:,:]
    
    #Para visualizar os conjuntos basta remover os comentários das próximas duas linhas
    #display(conjunto_treino_1)
    #display(conjunto_teste_2)
    
    
    #Utilizando o conjunto de treino 1 para gerar nossa árvore de decisão
    decision_tree_entropia = DT(criterion = 'entropy')
    
    #Agora determinaremos as colunas usadas durante a construção da árvore
    #Como queremos determinar que colunas geram um resultado melhor, teremos várias versões diferentes neste momento
    #Durante a interpretação dos dados, vimos que um dos dados que gerou certa ambiguidade foi o type, por isso, não utilizaremos esse dado
    colunas_treino = ['air_temperature_k', 'process_temperature_k','rotational_speed_rpm','torque_nm', 'tool_wear_min']   
   
    #Já a coluna a ser treinada não é modificada, uma vez que sempre queremos estimar 'random_failure'
    coluna_treinada = ['failure_type']
    
    #Separamos o conjunto de treino em 2, um com as colunas que serão usadas como entrada e outro com a coluna que será usada na saída
    conjunto_treino_1_a = conjunto_treino_1[colunas_treino]
    conjunto_treino_1_b = conjunto_treino_1[coluna_treinada]
    
    #Gerando a árvore de decisão
    decision_tree = gera_arvore(decision_tree_entropia,conjunto_treino_1_a,conjunto_treino_1_b)
    
    
    ##Testando a acurácia obtida
    
    #Assim como feito anteriormente, iremos dividir o conjunto de treino_2 em 2, mas dessa vez para uma finalidade diferente.
    #O conjunto de treino_2_a será a entrada da árvore, para que ela possa gerar a saída estimada
    conjunto_teste_2_a = conjunto_teste_2[colunas_treino]

    #O conjunto de treino_2_b será utilizado para medir a acurácia da nossa árvore de decisão, faremos isso comparando ele com a saída que árvore gerou
    conjunto_teste_2_b = conjunto_teste_2[coluna_treinada]
    
    #Agora veremos o valor obtido e sua acurácia
    predict = decision_tree.predict(conjunto_teste_2_a)
    print('Acuracia:',metrics.accuracy_score(conjunto_teste_2_b,predict))
    
    '''
        Podemos ver que a acuracia possui valor entre 97 e 98 por cento após algumas execuções, configurando um bom valor. Sendo assim, esses criterios serao mantidos na arvore
        de Decisão final criada para gerar a previsão dos resultados do caso de teste original
    '''
       
    
    ##Gerando Arquivo de Saida
    
    #Agora, visto que o método utilizado anteriormente gerou uma boa árvore, irei gerar uma nova árvore utilizando todo conjunto de treino para estimar a saida
    conjunto_treino_sem_random_a = conjunto_treino_sem_random[colunas_treino]
    conjunto_treino_sem_random_b = conjunto_treino_sem_random[coluna_treinada]
    decision_tree_final = gera_arvore(decision_tree_entropia,conjunto_treino_sem_random_a , conjunto_treino_sem_random_b)
    conjunto_teste = conjunto_teste[colunas_treino]
    decision_tree_resposta  = decision_tree_final.predict(conjunto_teste)
    
    #Caso queira ver a proporção da saida encontrada, basta remover o comentário da linha abaixo
    #display(Counter(decision_tree_resposta))
    
    saida = pd.DataFrame(decision_tree_resposta,columns = ['predictedValues'])
    
    #Transformando por fim a saida em arquivo csv
    saida.to_csv('predicted.csv', index_label = 'rowNumber',sep='\t')
    
    
    
# Executando a função main
if __name__ == '__main__':        
    main()
