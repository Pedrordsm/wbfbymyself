import cv2
import os

def plot_yolo_bboxes(img_path, txt_path, class_names=None, show_confidence=True):

    if not os.path.exists(img_path) or not os.path.exists(txt_path):
        print("Erro: Imagem ou arquivo de texto não encontrados.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("Erro: Não foi possível ler a imagem.")
        return

    h_img, w_img, _ = img.shape

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
              (0, 128, 128), (128, 0, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (0, 64, 64), (64, 0, 64), (192, 192, 192), (128, 128, 128), (0, 0, 0)]

    print(f"Encontrados {len(lines)} objetos.")

    for line in lines:
        parts = line.strip().split()
    
        class_id = int(parts[0])
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        width_norm = float(parts[3])
        height_norm = float(parts[4])
        score = float(parts[5]) 

        x_center = int(x_center_norm * w_img)
        y_center = int(y_center_norm * h_img)
        box_w = int(width_norm * w_img)
        box_h = int(height_norm * h_img)

        x1 = int(x_center - (box_w / 2))
        y1 = int(y_center - (box_h / 2))
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        color = colors[class_id % len(colors)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if class_names and class_id < len(class_names):
            label = class_names[class_id] + f" Score: {score:.2f}"
        else:
            label = f"ID {class_id} Score: {score:.2f}" 
        
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imwrite("resultado_orig_remove.jpg", img)

if __name__ == "__main__":
    imagem = "C:/Users/pedro/Downloads/0a4fbc9ade84a7abd1680eb8ba031a9d.png"
    arquivo_txt = "resultados_removeduplicatas/0a4fbc9ade84a7abd1680eb8ba031a9d.txt"
    nomes_classes = ['Aortic enlargement', 'Atelectasis', 'Cardiomegaly', 'Calcification', 'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration', 'Lung cavity', 'Lung cyst', 'Lung Opacity', 'Mediastinal shift', 'Nodule/Mass', 'Pulmonary fibrosis', 'Pneumothorax', 'Pleural thickening', 'Pleural effusion', 'Rib fracture', 'Other lesion']

    plot_yolo_bboxes(imagem, arquivo_txt, nomes_classes, show_confidence=True)