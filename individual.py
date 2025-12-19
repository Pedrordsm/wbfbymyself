import os
import pydicom

def get_dicom_dimensions(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path)
        return ds.Columns, ds.Rows
    except Exception as e:
        print(f"Erro ao ler {dicom_path}: {e}")
        return None, None

dicom_folder = 'dicomm'
output_dir = 'labels'
input_file = 'anotadores/R12.txt'
classes = [
    'Aortic enlargement', 'Atelectasis', 'Cardiomegaly', 'Calcification',
    'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA',
    'ILD', 'Infiltration', 'Lung cavity', 'Lung cyst', 'Lung Opacity',
    'Mediastinal shift', 'Nodule/Mass', 'Pulmonary fibrosis', 'Pneumothorax',
    'Pleural thickening', 'Pleural effusion', 'Rib fracture', 'Other lesion'
]

os.makedirs(output_dir, exist_ok=True)
dimensoes_cache = {}

try:
    with open(input_file, 'r') as f:
        linhas = f.readlines()

    print(f"Total de linhas no arquivo: {len(linhas)}")
    
    for idx, linha in enumerate(linhas):
        partes = [p.strip() for p in linha.strip().split(",")]
        
        # Ignora linhas vazias ou mal formadas
        if len(partes) < 6:
            continue
        
        image_id = partes[0]
        
        if image_id != "0a4fbc9ade84a7abd1680eb8ba031a9d": continue

        # Busca dimensões apenas se ainda não estiver no cache
        if image_id not in dimensoes_cache:
            dicom_path = os.path.join(dicom_folder, f"{image_id}.dicom")
            largura, altura = get_dicom_dimensions(dicom_path)
            
            if largura and altura:
                dimensoes_cache[image_id] = (largura, altura)
            else:
                print(f"Aviso: DICOM {image_id} não encontrado. Pulando...")
                continue
        
        IMG_WIDTH, IMG_HEIGHT = dimensoes_cache[image_id]
        
        # Coordenadas (últimos 4 valores)
        try:
            x_min, y_min, x_max, y_max = map(float, partes[-4:])
            # O nome da classe está entre o ID da imagem e as coordenadas
            class_name = partes[1] 
        except ValueError:
            continue

        if class_name not in classes:
            print(f"Classe desconhecida: '{class_name}'")
            continue
        
        class_id = classes.index(class_name)
        
        # Cálculo YOLO (Normalizado)
        x_center = ((x_min + x_max) / 2) / IMG_WIDTH
        y_center = ((y_min + y_max) / 2) / IMG_HEIGHT
        w = (x_max - x_min) / IMG_WIDTH
        h = (y_max - y_min) / IMG_HEIGHT
        
        # Garantir limites entre 0 e 1
        x_center, y_center = min(1.0, max(0.0, x_center)), min(1.0, max(0.0, y_center))
        w, h = min(1.0, max(0.0, w)), min(1.0, max(0.0, h))

        # Salva no arquivo TXT correspondente
        nome_arquivo = f'{image_id}R12.txt' # Geralmente o YOLO espera apenas o ID da imagem
        caminho_completo = os.path.join(output_dir, nome_arquivo)
        
        with open(caminho_completo, 'a') as out:
            out.write(f'{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} 1\n')

    print(f"\nSucesso! Labels salvos em '{output_dir}'.")
    print(f"Total de imagens únicas processadas: {len(dimensoes_cache)}")

except FileNotFoundError:
    print(f"Erro: O arquivo {input_file} não foi encontrado.")
except Exception as e:
    print(f"Erro inesperado: {e}")