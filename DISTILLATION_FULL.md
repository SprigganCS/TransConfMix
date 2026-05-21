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

### Peso tau (F1 micro de detecção vs GT, ou constante)

Por defeito (CONS2), $\tau$ é o **F1 score** entre as pseudo-detecções do teacher (após NMS em `x_s`) e o **ground truth de `x_s`** (`targets_s`), **agregado no batch inteiro** (micro: um único TP/FP/FN somando todas as imagens do batch).

Com **`--tau-const V`** (float), o mesmo peso **V** é usado em todos os batches — **não** se calcula F1; útil para ablações. $\gamma$ e o cálculo de $L_{cons}$ não mudam.

Com **uma classe** (`nc=1`), F1 micro e macro coincidem; o matching é **agnostico à classe** (não há filtro por `cls`).

Por imagem do batch:

1. Predições do teacher ordenadas por `conf` decrescente.
2. **Matching greedy**: cada predição emparelha o GT ainda não usado com maior IoU; se $\text{IoU} \geq 0{,}5$ conta como TP, senão FP.
3. GTs não emparelhados contam como FN.

\[
\tau = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}
\]

**Casos degenerados** (TP/FP/FN inteiros; $\tau = 0$ para não inflar $L_{cons2}$ quando não há sinal):

- Sem predições e sem GT: $\mathrm{TP}=\mathrm{FP}=\mathrm{FN}=0$.
- Sem predições, com GT: $\mathrm{FN}=|\mathrm{GT}|$, $\mathrm{TP}=\mathrm{FP}=0$.
- Com predições, sem GT: $\mathrm{FP}=|\mathrm{pred}|$, $\mathrm{TP}=\mathrm{FN}=0$.
- Denominador $2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}=0$: $\tau=0$.

O NMS do teacher continua com `conf_thres=0.25`, `iou_thres=0.5`; não há filtro extra de confiança só para o F1.

### Peso gamma (inalterado)

$\gamma$ continua sendo a fração de pseudo-boxes do ConfMix com `conf > 0.5` (mesma regra de antes; ver loop em `uda_train.py`).

### Exp 4 — peso por caixa do teacher (`--lcons2-box-weight`)

Alternativa ao escalar global $\tau$: cada detecção NMS do teacher em `x_s` recebe um peso $w_i$; dentro de `ComputeLoss`, as contribuições **lbox** e **lcls** nas atribuições positivas (âncoras) usam **média ponderada** $\sum_k w_k \ell_k / (\sum_k w_k + \varepsilon)$ por camada; **lobj** (grid inteiro) permanece como no YOLO.

Modos:

| Valor CLI | $w_i$ |
|-----------|--------|
| `teacher_conf` | confiança da caixa (0–1) |
| `teacher_iou_maxgt` | $\max_j \mathrm{IoU}(\mathrm{pred}_i, \mathrm{GT}_j)$ no mesmo `image_idx` |
| `teacher_conf_iou` | $\mathrm{conf}_i \times w_{\mathrm{iou}}$ |

Com `--lcons2-box-weight` **≠** `none`:

- $L_{total} = L_{det} + \gamma L_{cons} + L_{cons2}$ (sem fator global $\tau$ em $L_{cons2}$).
- `--tau-const` e o F1 para pesar $L_{cons2}$ são **ignorados**; `tau_batch.csv` **não** recebe linhas por batch (só header).
- Coluna `tau` em `tau_epoch.csv` e `w_bar` no tqdm: média de $w_i$ no batch (diagnóstico).

Default `none` = comportamento CONS2 ($\tau$ F1 ou `--tau-const`).

1. **Teacher** recebe `x_s` (imagem source original) com `pseudo=True`
2. Saída do teacher passa por **NMS** (conf_thres=0.25, iou_thres=0.5)
3. `output_to_target` produz `out_teacher` em **pixels** (xywh); **antes** de normalizar, o código calcula **τ** (F1 micro, IoU≥0.5 teacher vs GT no batch, ou `--tau-const`) ou pesos $w_i$ (Exp 4 / IoU), sempre nesta escala de pixels vs GT já normalizado.
4. Depois, monta-se `targets_teacher = [image_idx, class, x, y, w, h]` com xywh **dividido uma vez** por `w_img`/`h_img` (formato YOLO).
5. `compute_loss(pred_sp, targets_teacher, var_sp)` calcula a loss; com Exp 4, passa `target_weights=w_i`.
6. Contribuição na loss total: `tau * L_cons2` (modo CONS2) ou $L_{cons2}$ com pesos por caixa (Exp 4).

## Diferença em relação ao L_cons original

| Aspecto | L_cons (ConfMix) | L_cons2 (Teacher) |
|---|---|---|
| Input do modelo | student(confmix(x'_s, x_t)) | student(x'_s) |
| Targets | pseudo-labels de student source + teacher target, combinadas no mix | pseudo-labels do teacher em x_s |
| Peso | gamma (fração conf>0.5 nas pseudo do mix) | tau global ou `--tau-const` **ou** pesos $w_i$ (Exp 4) |
| Propósito | Consistência cross-domain no mix | Destilação de conhecimento teacher→student |

## Argumentos CLI

| Argumento | Descrição |
|---|---|
| `--use_distill` | Habilita L_cons2 com teacher |
| `--teacher_weights` | Caminho dos pesos do teacher |
| `--tau-const` | Opcional. Peso fixo para $\tau$ (ignora F1). Sem esta flag, $\tau$ = F1 por batch (CONS2). Incompatível com `--lcons2-box-weight` ≠ `none`. |
| `--lcons2-box-weight` | `none` (default), `teacher_conf`, `teacher_iou_maxgt`, `teacher_conf_iou` — Exp 4; ver secção acima. |

Sem `--tau-const`, tau é calculado **por batch** a partir do F1. Gamma continua automático como antes.

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

Com Exp 4, o postfix inclui `w_bar` (média de $w_i$ no batch) e `tau=0`.

```
L_det=0.045 L_cons=0.032 L_cons2=0.038 w_bar=0.61 tau=0
```

Caso contrário (CONS2):

```
L_det=0.045 L_cons=0.032 L_cons2=0.038 tau=0.72
```

No fim de cada época (CONS2, último batch):

```
L_det=0.042 L_cons=0.030 L_cons2=0.036 tau=0.70
```

Com Exp 4 o log de época usa `w_bar_mean=...` em vez de `tau`. A média por época continua em `tau_epoch.csv` na coluna `tau`.

## Arquivos `tau_batch.csv` e `tau_epoch.csv`

Com `--use_distill`, só **rank 0** (`RANK` 0 ou `-1` em single-GPU) grava em `save_dir/` (mesmo diretório que `hyp.yaml` / `results.csv`):

| Arquivo | Conteúdo |
|--------|-----------|
| `tau_batch.csv` | Uma linha por batch: `epoch`, `batch`, `tau`, `tp`, `fp`, `fn`, `n_pred`, `n_gt`, `gamma`. **Só modo CONS2 dinâmico** (sem `--tau-const` nem `--lcons2-box-weight`); com `--tau-const` ou Exp 4, sem linhas por batch (só header). |
| `tau_epoch.csv` | Uma linha por época: `epoch`, `ldet`, `gamma`, `lcons`, `tau`, `lcons2` — `ldet`/`lcons`/`lcons2` são as médias época das três losses (soma box+obj+cls de cada ramo, como no log `L_det`/`L_cons`/`L_cons2`); `gamma` e `tau` são **médias** dos valores por batch nessa época (no rank que gravou). Com `--tau-const`, a coluna `tau` coincide com esse escalar. Com Exp 4, `tau` é a média de $\bar w$ por batch (diagnóstico). |

Definições:

- **`n_pred`**: número de detecções do teacher **após NMS** no batch (`out_teacher`), as mesmas usadas no matching IoU vs GT.
- **`n_gt`**: número de GT de source no batch (`targets_s`).
- **`gamma` (por batch)**: fração de pseudo-labels do ConfMix nesse batch com confiança **> 0.5**, dividida pelo número de caixas em `targets_confmix` antes de truncar a coluna de conf — o mesmo escalar que pesa `L_cons` na loss total. No arquivo de época, `gamma` é a média desses valores por batch.

**`tau` (no CSV de época)**: média dos valores por batch — F1 (CONS2 dinâmico), `--tau-const`, ou $\bar w$ (Exp 4). TP/FP/FN por batch só em `tau_batch.csv` no modo CONS2 dinâmico.

**DDP:** com várias GPUs, o CSV reflete só os batches processados pelo **rank 0** (subconjunto da época). Single-GPU = época completa.

**Resume:** se `resume` e o arquivo já existir com dados, novas linhas são **append**; em treino novo (`not resume`), os CSVs são reiniciados com header.

## Comparação com versão anterior (KL denso)

| Aspecto | KL denso (anterior) | L_cons2 (atual) |
|---|---|---|
| Tipo de loss | KL binário + Smooth L1 | CIoU + BCE (standard YOLOv5) |
| Posições | Todas as anchors (~25k) | Apenas objetos detectados (pós-NMS) |
| Peso | lambda_obj, lambda_box, lambda_cls manuais | tau = F1 micro (teacher vs GT, IoU≥0.5) |
| Robustez a x_s ≠ x'_s | Fraca (box logits diferem) | Forte (CIoU tolera diferenças) |
| Complexidade | Alta (3 componentes, 3 hiperparâmetros) | Baixa (mesma compute_loss, sem hiperparâmetros extras) |

## Arquivos alterados

- `uda_train.py` — loop de treino, args, logs, `compute_tau_f1`, CSVs tau
