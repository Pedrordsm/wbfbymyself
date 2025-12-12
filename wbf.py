import os
import glob
from pathlib import Path

def ler_anotacao_yolo(caminho_arquivo_txt):
    nome_arquivo = Path(caminho_arquivo_txt).stem
    
    with open(caminho_arquivo_txt, 'r', encoding='utf-8') as f:
        linhas = f.readlines()
    
    anotacoes_arquivo = []
    
    for linha in linhas:
        linha = linha.strip()
        if linha:  
            partes = linha.split()
            if len(partes) >= 5:  
                anotacao = {
                    'id_classe': int(partes[0]),
                    'x_centro': float(partes[1]),
                    'y_centro': float(partes[2]),
                    'largura': float(partes[3]),
                    'altura': float(partes[4])
                }
                anotacoes_arquivo.append(anotacao)
    
    return (nome_arquivo, anotacoes_arquivo)

def calcular_iou(bbox1, bbox2):
    def yolo_para_coordenadas(bbox):
        x_centro, y_centro, largura, altura = bbox['x_centro'], bbox['y_centro'], bbox['largura'], bbox['altura']
        x1 = x_centro - largura / 2
        y1 = y_centro - altura / 2
        x2 = x_centro + largura / 2
        y2 = y_centro + altura / 2
        return x1, y1, x2, y2
    
    x1_1, y1_1, x2_1, y2_1 = yolo_para_coordenadas(bbox1)
    x1_2, y1_2, x2_2, y2_2 = yolo_para_coordenadas(bbox2)
    
    x1_intersecao = max(x1_1, x1_2)
    y1_intersecao = max(y1_1, y1_2)
    x2_intersecao = min(x2_1, x2_2)
    y2_intersecao = min(y2_1, y2_2)
    
    if x2_intersecao <= x1_intersecao or y2_intersecao <= y1_intersecao:
        return 0.0
    
    area_intersecao = (x2_intersecao - x1_intersecao) * (y2_intersecao - y1_intersecao)
    area_bbox1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_bbox2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    area_uniao = area_bbox1 + area_bbox2 - area_intersecao
    
    return area_intersecao / area_uniao if area_uniao > 0 else 0.0

def processar_txt_unico(tupla_arquivo, limiar_iou=0.5):
    nome_arquivo, lista_anotacoes = tupla_arquivo
    
    # se não tem anotação retorna listas vazias
    if not lista_anotacoes:
        return [], [], []
    
    boxes = []    
    scores = []
    labels = []
    processadas = set()
    
    # itera para cada anotação
    for i, anotacao_base in enumerate(lista_anotacoes):
        # se na lista de processadas, pula
        if i in processadas:
            continue
            
        # cluster para anotações redundantes
        grupo_redundantes = [anotacao_base]
        indices_grupo = [i]
        processadas.add(i)
        
        # procura por anotações redundantes
        for j, anotacao_poss in enumerate(lista_anotacoes[i+1:], i+1):
            # se já processada, pula
            if j in processadas:
                continue
                
            # verifica classe e iou - se iou > limiar adiciona ao grupo
            if (anotacao_base['id_classe'] == anotacao_poss['id_classe'] and 
                calcular_iou(anotacao_base, anotacao_poss) > limiar_iou):
                
                grupo_redundantes.append(anotacao_poss)
                indices_grupo.append(j)
                processadas.add(j)
        
        # se só tem uma anotação, adiciona com peso 1.0
        if len(grupo_redundantes) == 1:
            anotacao = grupo_redundantes[0]
            
            # coordenadas de canto
            x1 = anotacao['x_centro'] - anotacao['largura'] / 2
            y1 = anotacao['y_centro'] - anotacao['altura'] / 2
            x2 = anotacao['x_centro'] + anotacao['largura'] / 2
            y2 = anotacao['y_centro'] + anotacao['altura'] / 2
            
            boxes.append([x1, y1, x2, y2])
            scores.append(1.0)  
            labels.append(anotacao['id_classe'])
            
        else:
            # calcula e guarda anotação média em relação às redundantes
            anotacao_media = {
                'id_classe': grupo_redundantes[0]['id_classe'],
                'x_centro': sum(a['x_centro'] for a in grupo_redundantes) / len(grupo_redundantes),
                'y_centro': sum(a['y_centro'] for a in grupo_redundantes) / len(grupo_redundantes),
                'largura': sum(a['largura'] for a in grupo_redundantes) / len(grupo_redundantes),
                'altura': sum(a['altura'] for a in grupo_redundantes) / len(grupo_redundantes)
            }
            
            # para cada anotação 
            for x, anotacao_original in enumerate(grupo_redundantes):
                # calcula IoU desta anotação com a média do grupo
                iou_com_media = calcular_iou(anotacao_original, anotacao_media)
                
                # converte anotação original para formato de coordenadas de canto
                x1 = anotacao_original['x_centro'] - anotacao_original['largura'] / 2
                y1 = anotacao_original['y_centro'] - anotacao_original['altura'] / 2
                x2 = anotacao_original['x_centro'] + anotacao_original['largura'] / 2
                y2 = anotacao_original['y_centro'] + anotacao_original['altura'] / 2
            
                boxes.append([x1, y1, x2, y2])
                scores.append(iou_com_media) 
                labels.append(anotacao_original['id_classe'])
                
    return boxes, scores, labels
