# L_cons2: Teacher Consistency Loss

## Motivação

A abordagem anterior usava KL divergência densa entre teacher e student (sobre objectness, box regression e class score em todas as posições de anchor). Essa formulação tinha dois problemas:

1. O KL denso tenta alinhar representações internas em ~25k posições, incluindo background, gerando ruído
2. Teacher e student veem imagens diferentes (x_s vs x'_s), tornando o alinhamento pixel-a-pixel de box regression contraproducente

A nova abordagem substitui o KL denso por uma **segunda consistency loss (L_cons2)** que usa pseudo-labels do teacher como supervisão, calculada via `compute_loss` — a mesma loss function do YOLOv5 (CIoU + BCE obj + BCE cls).

## Formulação

### Perda total

$$L_{total} = L_{det} + \gamma \cdot L_{cons} + \tau \cdot L_{cons2}$$

onde:

- **L_det** = `compute_loss(student(x'_s), GT_source)` — loss supervisionada com ground truth
- **L_cons** = `compute_loss(student(confmix), pseudo_labels_mix)` — consistency no ConfMix (inalterado)
- **L_cons2** = `compute_loss(student(x'_s), pseudo_labels_teacher)` — consistency com teacher

### Componentes de L_cons2

`compute_loss` calcula internamente:

$$L_{cons2} = h_{box} \cdot L_{box}^{CIoU} + h_{obj} \cdot L_{obj}^{BCE} + h_{cls} \cdot L_{cls}^{BCE}$$

onde $h_{box}, h_{obj}, h_{cls}$ são os hyperparâmetros do YOLOv5 (mesmos usados em L_det e L_cons).

### Peso tau

$$\tau = \frac{|\{d \in D_T : \text{conf}(d) > 0.5\}|}{|D_T|}$$

onde $D_T$ são as pseudo-detecções do teacher após NMS. Calculado identicamente ao $\gamma$ do L_cons original.

## Fluxo de dados

1. **Teacher** recebe `x_s` (imagem source original) com `pseudo=True`
2. Saída do teacher passa por **NMS** (conf_thres=0.25, iou_thres=0.5)
3. Pseudo-labels são formatadas como targets `[image_idx, class, x, y, w, h]` normalizados
4. **tau** é calculado como fração de detecções com confiança > 0.5
5. `compute_loss(pred_sp, targets_teacher, var_sp)` calcula a loss
6. Contribuição na loss total: `tau * L_cons2`

## Diferença em relação ao L_cons original

| Aspecto | L_cons (ConfMix) | L_cons2 (Teacher) |
|---|---|---|
| Input do modelo | student(confmix(x'_s, x_t)) | student(x'_s) |
| Targets | pseudo-labels de student source + teacher target, combinadas no mix | pseudo-labels do teacher em x_s |
| Peso | gamma | tau |
| Propósito | Consistência cross-domain no mix | Destilação de conhecimento teacher→student |

## Argumentos CLI

| Argumento | Descrição |
|---|---|
| `--use_distill` | Habilita L_cons2 com teacher |
| `--teacher_weights` | Caminho dos pesos do teacher |

Tau é calculado automaticamente a cada batch, sem parâmetro manual.

### Exemplo de uso

```bash
python uda_train.py \
  --data data/Sim10K2Cityscapes.yaml \
  --weights runs/source_only/best.pt \
  --teacher_weights runs/source_only/best.pt \
  --use_distill \
  --epochs 50 --batch-size 2 --img 600
```

## Logs

Durante o treinamento, a barra de progresso mostra:

```
L_det=0.045 L_cons=0.032 L_cons2=0.038 tau=0.85
```

No fim de cada época:

```
L_det=0.042 L_cons=0.030 L_cons2=0.036 tau=0.87
```

## Comparação com versão anterior (KL denso)

| Aspecto | KL denso (anterior) | L_cons2 (atual) |
|---|---|---|
| Tipo de loss | KL binário + Smooth L1 | CIoU + BCE (standard YOLOv5) |
| Posições | Todas as anchors (~25k) | Apenas objetos detectados (pós-NMS) |
| Peso | lambda_obj, lambda_box, lambda_cls manuais | tau automático (fração de conf > 0.5) |
| Robustez a x_s ≠ x'_s | Fraca (box logits diferem) | Forte (CIoU tolera diferenças) |
| Complexidade | Alta (3 componentes, 3 hiperparâmetros) | Baixa (mesma compute_loss, sem hiperparâmetros extras) |

## Arquivos alterados

- `uda_train.py` — loop de treino, args, logs
