# Classificação de Lesões de Pele com IA

Este projeto aplica redes neurais convolucionais para classificar automaticamente lesões de pele usando a base HAM10000.

## Como usar
1. Instale as dependências:
```
pip install -r requirements.txt
```
2. Baixe a base HAM10000 no Kaggle e coloque em `data/`.
3. Execute os scripts na ordem:
   - `scripts/preprocess.py`
   - `scripts/train.py`
   - `scripts/evaluate.py`
