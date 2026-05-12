# Verificação da formulação matemática do L_KL

## Veredicto geral: parcialmente correto

A fórmula KL binária e os índices estão certos, mas a **agregação** não reflete o que o código faz.

---

## 1. Índice de âncora `a`

**Correto.** O tensor por escala tem shape `[bs, na, ny, nx]` após extrair `[..., 4]`. O índice `a` na formulação é necessário e fiel.

---

## 2. Agregação como `(1/N) * sum` — INCORRETA

O código **não** faz uma média global única. Faz em dois passos:

1. Para cada escala `s`: `kl.mean()` → média sobre `(batch, anchor, i, j)` → um escalar
2. Soma os escalares de todas as escalas e divide por `S` (número de escalas)

Referência: `uda_train.py` linhas 585–588:
```python
kl_total += kl.mean()                            # mean per scale
loss_kl = kl_total / max(len(pred_teacher), 1)   # divide by S
```

Isso é equivalente a:

$$L_{KL} = \frac{1}{S}\sum_{s=1}^{S} \left[\frac{1}{N_s}\sum_{b,a,i,j} D_{KL}\!\left(p_T^{b,i,j,a,s} \| p_S^{b,i,j,a,s}\right)\right]$$

onde $N_s = B \times n_a \times H_s \times W_s$.

**Diferença prática:** na fórmula proposta (média global `1/N`), escalas maiores (P3, 80×80) dominam. No código (média por escala, depois média entre escalas), cada escala contribui igualmente — P5 (20×20) pesa tanto quanto P3 (80×80).

---

## 3. KL elemento a elemento

**Correto.** Operação via broadcasting sobre `[bs, na, ny, nx]`.

---

## 4. Detalhes omitidos na formulação

### a) Dimensão de batch `b`

O `.mean()` inclui o batch. A fórmula proposta tem `sum_{s,a,i,j}` mas não menciona `b`. Deve-se ou explicitar o índice `b`, ou dizer que a expressão é "per sample" e a média sobre o batch é implícita.

### b) Clamping com eps

As probabilidades são clampadas para `[eps, 1-eps]` (eps=1e-4) antes do KL. Não precisa entrar na fórmula formal, mas pode ser mencionado em texto como detalhe de estabilidade numérica.

---

## 5. Correção sugerida

### Opção A (explícita)

$$L_{KL} = \frac{1}{S}\sum_{s=1}^{S} \left[\frac{1}{N_s}\sum_{b,a,i,j} D_{KL}\!\left(p_T^{b,i,j,a,s} \| p_S^{b,i,j,a,s}\right)\right]$$

Com definição: "$S$ denotes the number of detection scales and $N_s = B \times n_a \times H_s \times W_s$ is the total number of objectness predictions at scale $s$, where $B$ is the batch size."

### Opção B (compacta)

$$L_{KL} = \frac{1}{S}\sum_{s=1}^{S} \mathbb{E}_{b,a,i,j}\!\left[D_{KL}\!\left(p_T^{b,i,j,a,s} \| p_S^{b,i,j,a,s}\right)\right]$$

Com texto explicando que a expectativa é sobre todas as posições, anchors e amostras do batch em cada escala, e que a média é tomada uniformemente entre escalas.
