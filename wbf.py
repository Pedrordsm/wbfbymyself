import os
import glob
from pathlib import Path

def ler_anotacoes_yolo(caminho_pasta):

    anotacoes = {}

    arquivos_txt = glob.glob(os.path.join(caminho_pasta, "*.txt"))
    
    for arquivo_txt in arquivos_txt:
        # tira .txt
        nome_arquivo = Path(arquivo_txt).stem

        with open(arquivo_txt, 'r', encoding='utf-8') as f:
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
        
        anotacoes[nome_arquivo] = anotacoes_arquivo
    
    return anotacoes

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

def processar_redundancia(anotacoes, limiar_iou=0.5):
    resultado = {}
    
    for nome_arquivo, lista_anotacoes in anotacoes.items():
        anotacoes_finais = []
        pesos = []
        
        processadas = set()
        
        for i, anotacao1 in enumerate(lista_anotacoes):
            if i in processadas:
                continue
            grupo = [anotacao1]
            processadas.add(i)
            
            for j, anotacao2 in enumerate(lista_anotacoes[i+1:], i+1):
                if j in processadas:
                    continue
                if (anotacao1['id_classe'] == anotacao2['id_classe'] and 
                    calcular_iou(anotacao1, anotacao2) > limiar_iou):
                    grupo.append(anotacao2)
                    processadas.add(j)
            
            anotacao_media = {
                'id_classe': grupo[0]['id_classe'],
                'x_centro': sum(a['x_centro'] for a in grupo) / len(grupo),
                'y_centro': sum(a['y_centro'] for a in grupo) / len(grupo),
                'largura': sum(a['largura'] for a in grupo) / len(grupo),
                'altura': sum(a['altura'] for a in grupo) / len(grupo)
            }
            
            peso = len(grupo)
            
            anotacoes_finais.append(anotacao_media)
            pesos.append(peso)
        
        resultado[nome_arquivo] = {
            'anotacoes': anotacoes_finais,
            'pesos': pesos
        }
    
    return resultado