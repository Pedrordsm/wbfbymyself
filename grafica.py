import matplotlib.pyplot as plt
import numpy as np
import warnings
from collections import Counter, defaultdict
from pathlib import Path
warnings.filterwarnings('ignore')

# função para gerar o gráfico de distribuição das classes
def geragrafico(caminho_arquivo, nome):
    # carregar os dados do arquivo
    data = np.loadtxt(caminho_arquivo, delimiter=',', dtype=str)

    # estruturas para armazenar informações
    images_with_classes = set()
    class_distribution = Counter()
    images_per_class = defaultdict(set)

    # processar cada linha do arquivo
    for row in data:
        # pegar o ID da imagem e o nome da classe
        image_id = row[0]
        class_name = row[1]
        
        # se a classe não for vazia, atualizar as contagens
        if class_name != '':
            # adicionar a imagem ao conjunto de imagens com classes
            images_with_classes.add(image_id)
            # atualizar a distribuição de classes
            class_distribution[class_name] += 1
            # adicionar a imagem ao conjunto de imagens para a classe específica
            images_per_class[class_name].add(image_id)

    # se houver dados para plotar
    if class_distribution:
        # criar o gráfico de barras
        plt.figure(figsize=(10, 5))
        
        # pega a lista de classes e suas contagens
        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())
        # total de imagens com pelo menos uma classe anotada
        total_images = len(images_with_classes)

        # plotar o gráfico de barras
        bars = plt.bar(classes, counts, color='skyblue', edgecolor='black')
        plt.xlabel('Classes', fontsize=12)
        plt.ylabel('Número de Anotações', fontsize=12)
        plt.title('Distribuição das Classes do Radiologista '+ nome + " em " + str(total_images) + " imagens", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # para cada barra, adicionar o valor no topo, o zip une barras e contagens
        for bar, count in zip(bars, counts):
            # adicionar o texto no topo da barra
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(count), ha='center', va='bottom')
        
        # ajustar layout e salvar o gráfico
        plt.tight_layout()
        plt.savefig('distribuicao_classes_' + nome + '.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    caminho_pasta = Path('anotadores')
    for caminho_arquivo in caminho_pasta.glob('*.txt'):
        nome = caminho_arquivo.stem
        geragrafico(caminho_arquivo, nome)