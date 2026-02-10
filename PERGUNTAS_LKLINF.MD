# Verificação definitiva do L_kl (KL objectness) — Instrumentação e critérios

## Objetivo
Garantir que:
1) `L_kl` nunca vira `inf`/`nan` em execução real (50 epochs)
2) o guard `if not isfinite(L_kl): L_kl=0` NÃO está mascarando problema recorrente
3) o clamp/FP32 estão realmente ativos durante treino (com AMP habilitado)

---

## 1) Contadores e logging por epoch (obrigatório)
Adicionar contadores acumulados no loop de treino (reset por epoch):
- `kl_nonfinite_steps` (int)
- `kl_clamp_hits` (int)  # opcional: quantas vezes pT/pS estava fora do intervalo antes do clamp
- `pT_min_epoch`, `pT_max_epoch`, `pS_min_epoch`, `pS_max_epoch` (float)
- `L_kl_mean_epoch` (float)  # média do KL ao longo do epoch (antes do weight lambda)

**Implementação sugerida:**
- A cada batch, após calcular `pT`/`pS` (já em FP32):
  - atualizar min/max com `min()`/`max()` (com `.item()` e sem sync excessivo; pode fazer só a cada N iters)
  - atualizar `L_kl_running += L_kl.detach()`

Ao final do epoch, logar uma linha tipo:
- `KL: mean=..., nonfinite_steps=..., pT=[min,max], pS=[min,max]`

---

## 2) Guard explícito + warning com taxa limitada
Manter o guard existente, mas:
- incrementar `kl_nonfinite_steps += 1`
- printar warning **apenas nas primeiras K ocorrências** (ex.: K=5) para não poluir log

---

## 3) Ajuste do eps (confirmar e padronizar)
Padronizar `eps` para ser robusto em AMP:
- usar `eps = 1e-4` (não 1e-6)

E deixar explícito no código com comentário:
> eps larger to avoid saturation under AMP/FP16.

---

## 4) Assert opcional (modo debug)
Adicionar flag `--debug_kl` (default False). Se True:
- se `kl_nonfinite_steps > 0` em qualquer epoch → `raise RuntimeError`
- opcional: assert de shapes `pred_T[i].shape == pred_Sp[i].shape` em i==0 do primeiro batch

---

## 5) Critérios de aceitação (para considerar “corrigido de vez”)
Após rodar pelo menos 1 epoch completo:
- `kl_nonfinite_steps == 0`  (ideal)
Se não der para zerar:
- `kl_nonfinite_steps` deve ser raríssimo (ex.: < 0.1% dos steps) e com log de ranges mostrando saturação extrema.

E após 50 epochs:
- `kl_nonfinite_steps` não deve crescer continuamente (sinal de instabilidade).

---

## 6) Checklist final (responder com evidência)
Peço que você responda com:
- Trecho final do bloco KL (linhas relevantes)
- Um exemplo de log por epoch contendo:
  - `L_kl_mean_epoch`
  - `kl_nonfinite_steps`
  - ranges `pT` e `pS`
- Confirmação do valor de `eps` usado

Respostas:
- Trecho do KL: ver [uda_train.py](uda_train.py#L552-L588).
- Exemplo de log por epoch: nao disponivel porque nao foi permitido iniciar o treinamento aqui. O formato esperado e:
  "KL: mean=... nonfinite_steps=... clamp_hits=... pT=[min,max] pS=[min,max]".
- Valor de `eps`: 1e-4 (comentado no codigo). Ver [uda_train.py](uda_train.py#L566).
