import os
import glob
import numpy as np
from pathlib import Path
from wbf_anota_mediaa import *

# normalizar de para 0 e 1 as coordenadas
def normalizar_coordenadas(boxes):
    boxes_normalizadas = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        
        if x2 <= x1:
            x2 = x1 + 0.001
        if y2 <= y1:
            y2 = y1 + 0.001
            
        boxes_normalizadas.append([x1, y1, x2, y2])
    
    return boxes_normalizadas
'''
# aplicar WBF
def aplicar_wbf_final(boxes_list, scores_list, labels_list, iou_thr=0.5, skip_box_thr=0.0001):
    try:
        from ensemble_boxes import weighted_boxes_fusion
        
        if not boxes_list:
            return [], [], []
        
        boxes_list_norm = []
        for boxes in boxes_list:
            boxes_norm = normalizar_coordenadas(boxes)
            boxes_list_norm.append(boxes_norm)
        
        weights = [1.0] * len(boxes_list_norm)

        
        boxes_finais, scores_finais, labels_finais = weighted_boxes_fusion(
            boxes_list_norm, 
            scores_list, 
            labels_list, 
            weights=weights, 
            iou_thr=iou_thr, 
            skip_box_thr=skip_box_thr
        )
        
        return boxes_finais, scores_finais, labels_finais
        
    except ImportError:
        return [], [], []
    except Exception as e:
        return [], [], []
'''
def salvar_resultado_yolo(boxes, scores, labels, arquivo_saida):
    try:
        with open(arquivo_saida, 'w') as f:
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                x_centro = (x1 + x2) / 2
                y_centro = (y1 + y2) / 2
                largura = x2 - x1
                altura = y2 - y1
                
                f.write(f"{int(label)} {x_centro:.6f} {y_centro:.6f} {largura:.6f} {altura:.6f} {score:.6f}\n")
        print(f"Resultado salvo em: {arquivo_saida}")
        
    except Exception as e:
        print(f"Erro ao salvar: {e}")

def processar_arquivo_individual(caminho_arquivo, pasta_saida, limiar_iou=0.5, iou_thr_wbf=0.5):
    try:
        tupla_arquivo = ler_anotacao_yolo(caminho_arquivo)
        nome_arquivo, anotacoes_originais = tupla_arquivo
        
        if not anotacoes_originais:
            print(f"{nome_arquivo}: arquivo vazio")
            return False
    
        boxes, scores, labels = processar_txt_unico(tupla_arquivo, limiar_iou)
        
        if not boxes:
            print(f"{nome_arquivo}: sem anotações após processamento")
            return False
        
        boxes_norm = normalizar_coordenadas(boxes)
        '''
        boxes_finais, scores_finais, labels_finais = aplicar_wbf_final(
            [boxes_norm], [scores], [labels], iou_thr_wbf, 0.0001
        )
        
        
        if len(boxes_finais) == 0:
            print(f"{nome_arquivo}: WBF falhou")
            return False
        '''
        arquivo_saida = os.path.join(pasta_saida, f"{nome_arquivo}.txt")
        salvar_resultado_yolo(boxes, scores, labels, arquivo_saida)
        return True
        
    except Exception as e:
        print(f"{nome_arquivo}: erro - {e}")
        return False

def main():
    caminho_pasta ="C:/Users/pedro/OneDrive/Desktop/labels/train"
    pasta_saida = "resultados_removeduplicatas" 
    caminho_pasta ="C:/Users/pedro/OneDrive/Desktop/labels/train"
    pasta_saida = "resultados_removeduplicatas" 
    limiar_iou = 0.5              
    iou_thr_wbf = 0.5           
    

    os.makedirs(pasta_saida, exist_ok=True)
    
    arquivos_txt = glob.glob(os.path.join(caminho_pasta, "*.txt"))
    
    if not arquivos_txt:
        return
    
    sucessos = 0
    falhas = 0
    
    for i, arquivo_txt in enumerate(arquivos_txt, 1):
        nome_arquivo = Path(arquivo_txt).stem
        print(f"\n[{i}/{len(arquivos_txt)}] Processando: {nome_arquivo}")
        if not nome_arquivo.strip() == "0a4fbc9ade84a7abd1680eb8ba031a9d":
            continue    
        if not nome_arquivo.strip() == "0a4fbc9ade84a7abd1680eb8ba031a9d":
            continue    
        if processar_arquivo_individual(arquivo_txt, pasta_saida, limiar_iou, iou_thr_wbf):
            sucessos += 1
        else:
            falhas += 1

    print(f"\nProcessamento concluído: {sucessos} sucessos, {falhas} falhas.")

if __name__ == "__main__":
    main()