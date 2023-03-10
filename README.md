# Indicium DataScience - Cientista de Dados

<h2>Proposta</h2>
O propósito deste arquivo é fazer a análise de dados do arquivo 'desafio_manutencao_preditiva_treino' e, com o uso de Machine learning conseguir gerar uma previsão de resultados para um arquivo chamado de 'desafio_manutencao_preditiva_teste. A análise de dados e o processo de Machine Learning foram realizados utilizando a linguagem Python3 no Jupyter notebook.

<h2>Execução</h2>
Para executar o programa, primeiro é necessário baixar e extrair os arquivos. Após isso, basta abrir a pasta em algum programa para execução como visual studio code. O primeiro passo antes de executar é digitar 'pip install -r requirements.txt' no seu prompt de comando uma vez no diretório correto para instalar os pacotes necessário para execução. Os arquivos de dados e o relatório de EDA são apenas para visualização, mas caso queira executar novamente o relatório devido a algum erro, basta executar o arquivo 'EDA_report_generate.py'. O arquivo 'Desafio.ipynb' é o principal e é lá que é feito o tratamento de dados, visualização e é gerado o arquivo de saída. Devido a isso seu funcionamento será melhor explicado em partes a seguir.

<h2>Busca de dados</h2>
Para realizar o processo, o trabalho foi subdivido em diferentes partes. Durante a primeira parte, é feita a leitura dos dados desejados utilizando a biblioteca Pandas do Python. Esses dados foram salvos em duas variáveis diferentes, referentes aos arquivos.

<h2>Tratamento de dados</h2>
Agora que nossos dados foram salvos, o tratamento deles será iniciado. Antes de fazer o tratamento, entretanto, foi necessária uma visualização dos dados que foi possível através de um relatório de EDA (exploratory data analysis) que se encontra junto desse arquivo. Este relatório é gerado como página da web para que possa manter suas propriedades e permitir uma melhor visualização dos dados. Através dele, podemos observar que variáveis possuem correlação com outras, a presença ou ausência de linhas e colunas vazias, além de que colunas são apenas identificadores e quais representam dados mais importantes. Esse relatório foi gerado através biblioteca dataprep. Uma vez observados esses padrões, as colunas que foram vistas como apenas identificadores foram desconsideradas, uma vez que não irão gerar informações relevantes para os próximos passos do nosso programa.

<h2>Análise de dados</h2>
Neste passo, com o auxílio da bilioteca pyplot iremos gerar gráficos para ajudara visualizar que informações são mais relevantes para o nosso programa. As informações foram geradas como gráficos utilizando failute_type que é a coluna que queremos ter o resultado previsto como principal foco de oservação. Para isso, veremos como cada coluna da nossa base de dados se comporta com relação a coluna failure_type. O gráfico gerado permite que vejamos os diferentes tipos de erro, mas é recomendada cautela, uma vez que os dados absolutos podem soar contra intuitivos as vezes, por isso, é importante ter em mente a proporção dos dados durante o processo de análise dos gráficos. Por fim, vale ressaltar que os gráficos são gerados como páginas da web, para que possam manter suas finalizades de interatividade e que um dos tipos de erro foi removido da base de dados por ser gerado aleatoriamente fazendo com que não tenha padrões e portanto é mais prejudicial do que benéfico a nossa base de dados.

<h2>Método escolhido</h2>
Após realizada a análise gráfica, iremos utilizar uma árvore de decisão que servirá como classificador do nosso problema. Entretando, antes de gerarmos nossa árvore definitiva, utilizaremos uma árvore teste para medir a acurácia do nosso problema. Como queremos estimar sua acurácia, precisamos saber quais resultados eram esperados e quais foram encontrados pela árvore, por isso, dividiremos nosso conjunto de treino em 2. De maneira análoga a como o conjunto de treino e de teste funcionam no problema original, 2/3 do nosso conjunto de treino seram um novo conjunto de treino, enquanto 1/3 será usado como conjunto de teste.
Após isso, nossa árvore será gerada com a ajuda da biblioteca sklearn do Python. A árvore de decisão recebe como entrada um conjunto de treino com as colunas que são parâmetros e a coluna que ela deverá estimar. Através disso, ela criará um processo interno de decisão, e agora, será capaz de estimar a coluna desejada recebendo apenas as colunas de entrada como parâmetro. Durante esse passo, é importante ressaltar que a coluna type deixou de ser utilizada uma vez que é a coluna que pareceu ser menos eficiente na análise numérica, além de não ser a melhor para a árvore de decisão, pois seria necessária uma adaptação de type para valores numéricos.
Com isso em mente, dividimos nossa nova base de dados de treino com as colunas de treino e a coluna que é treinada e através da função gera_arvore criamos finalmente nossa árvore de decisão.

<h2>Teste de acurácia</h2>
Para ver o quão eficiente é a nossa árvore de decisão, utilizamos o novo conjunto de testepara observar a acurácia das previsões da árvore de decisão. Para isso, passamos como entrada o  conjunto de teste com a coluna de treino e vemos através do método metrics da biblioteca sklearn qual a acurácia que obtemos com essa previsão, comparando-a com a base de dados correta. Executando algumas vezes, percebemos que nossa árvore conseguiu cerca de 98% de acurácia durante sua previsão, portanto, os critérios que escolhemos parecem ter sido bons e podemos seguir para o último passo.

<h2>Arquivo de saída</h2>
Iremos gerar o arquivo de saída depois de prevermos o resultado do nosso conjunto de teste original. Para isso, iremos recriar uma árvore de decisão com os mesmos critérios utilizados durante nosso experiemento anterior, mas, utilizando dessa vez todo o conjunto de treino que tinhamos originalmente. Após isso, iremos utilizá-la para gerar a estimativa do conjunto de teste, e por fim, utilizando o método to_csv da bibliotecas Pandas exportaremos esse arquivo para Csv. O arquivo Csv também se encontra em anexo.

<h2>Arquivos</h2>
O arquivo 'desafio.ipynb' foi feito em Python utilizando jupyter notebook. Lá está contido o código principal, além de explicações acerca de cada parte importante do código.
'EDA_report_generate.py' é um arquivo python contendo o código utilizado para gerar o relatório EDA. Este relatório é o arquivo 'Relatorio_de_EDA.html'.
Os arquivos.
Os arquivos 'desafio_manutencao_preditiva_teste.csv' e 'desafio_manutencao_preditiva_treino.csv' foram as bases de dado utilizadas.
'predicted.csv' é o arquivo de saída, contendo as previsões de tipo de falha do conjunto de treino.
Por fim, 'requirements.txt' possui todos os pacotes utilizados pelo programa.

<h2>Observações importantes</h2>
Durante todo o programa, podem ser vistas linhas de código comentadas. Essas linhas tem como propósito permitir que os dados sejam visualizados, ou, permitir que os arquivos como relatório de EDA ou gráficos sejam gerados no seu computador. Também foram utilizadas duas funções simples, gera_arvore, que recebe uma árvore de decisão e a base de dados com as colunas de treino e colunas a serem treinadas e gera a árvore de decisão e busca_arquivos que recebe o caminho de um arquivo csv e retorna esse arquivo. Vale ressaltar que em python, precisamos usar '\\' e não apenas '\' durante a escrita do caminho para evitar erros de execução.
