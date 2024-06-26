#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:57:21 2024

@author: teodoro
"""

import numpy as np

entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
saidas = np.array([0, 1, 1, 1])
pesos = np.array([0.0, 0.0])
taxaAprendizagem = 0.1


def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0
def calculasaida(registro):
    s = registro.dot(pesos)
    return stepFunction(s)

def treinar():
    erroTotal = 1
    while(erroTotal!=0):
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculasaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro
            for j in range (len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas [i] [j] * erro)
                print('pesos atualizados: ' + str (pesos[j]))
        print('Total de erros: ' + str(erroTotal)) 
            

treinar()
print('rede neural treinada!')
for i in range (len(entradas)):
    print(calculasaida(entradas[i]))







        