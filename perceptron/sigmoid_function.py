#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:28:55 2024

@author: teodoro
"""

import numpy as np

def sigmoid(soma):
    return 1/(1+ np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1- sig)

a = sigmoid(0.5)
b = sigmoidDerivada(a)
entradas = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])

saidas = np.array([[0], [1], [1], [0]])



pesos0 = 2  * np.random.random((2, 3)) - 1
pesos1 = 2* np.random.random((3 , 1 )) - 1

epocas = 1000000
taxaAprendizagem = 0.6
momento = 1

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro:" + str(erroCamadaSaida))
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    pesos1trans = pesos1.T
    deltaSaidaPeso = deltaSaida.dot(pesos1trans)
    deltaCamadaOculta = deltaSaidaPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTrans = camadaOculta.T
    pesos1Novo  = camadaOcultaTrans.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesos1Novo * taxaAprendizagem)
    
    camadaEntradaTrans = camadaEntrada.T
    pesosNovo0 = camadaEntradaTrans.dot(deltaCamadaOculta)
    pesos0  = (pesos0 * momento ) + (pesosNovo0* taxaAprendizagem )
    

