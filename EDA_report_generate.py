import pandas as pd
from dataprep.eda import create_report

#Função responsável por receber o caminho dos arquivos CSV desejados e retorná-los 
def busca_arquivos(caminho_csv):
    return pd.read_csv(caminho_csv)

def main():

    conjunto_treino = busca_arquivos('desafio_manutencao_preditiva_treino.csv')
    conjunto_teste = busca_arquivos('desafio_manutencao_preditiva_teste.csv')
        
    #Utilizando a bilioteca dataprep do Python, irei gerar um relatório de EDA para nossa base de dados
    #O report gerado está em anexo, mas caso queira gerar ele novamente, basta tirar o comentário da linha abaixo
    create_report(conjunto_treino, title = 'Relatorio  de EDA').show_browser()
    
# Executando a função main
if __name__ == '__main__':        
    main()
