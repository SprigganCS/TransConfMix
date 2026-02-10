# Implementação: KL(Objectness) focado por máscara do teacher (evitar domínio do background)

## Motivação (1 linha)
O KL denso está sendo dominado por background (clamp_hits muito alto), o que tende a reduzir recall e derrubar mAP@0.5. Vamos aplicar o KL **apenas** em regiões onde o teacher indica evidência de objeto, sem virar pseudo-labeling.

---

## Objetivo técnico
Substituir o cálculo atual de `L_kl` por uma versão **masked/weighted** usando o mapa de objectness do teacher `pT` como máscara.

- Mantém:
  - teacher(xs) vs student(xs') (KL entre domínios)
  - KL somente no objectness
  - FP32 fora do autocast, clamp, guard finitude
- Altera:
  - KL deixa de ser computado em todo o mapa; passa a ser computado **apenas onde o teacher “acende”**

---

## Mudanças necessárias

### 1) Adicionar flags
Adicionar no parser (ou reaproveitar config):
- `--kl_mask_mode` com opções: `hard`, `soft`, `none` (default: `hard`)
- `--kl_tau` (default: `0.01`)
- `--kl_mask_power` (default: `1.0`)  # usado só no modo soft

**Sem isso, implemente hard com tau=0.01 fixo**, mas preferível expor flags.

---

### 2) Implementar máscara e KL mascarado (substituir bloco atual do KL)
Dentro do bloco FP32 do KL (onde você já tem `pT` e `pS` clampados), implemente:

#### Definições
- `pT`: sigmoid(obj_teacher), FP32, clamp(eps, 1-eps), detach
- `pS`: sigmoid(obj_student), FP32, clamp(eps, 1-eps)

- `KL_map`: KL binário por elemento:
  - `KL_map = pT*log(pT/pS) + (1-pT)*log((1-pT)/(1-pS))`

#### Modo HARD (recomendado para começar)
- Máscara binária:
  - `M = (pT > kl_tau).float()`
- Loss normalizada pelos ativos:
  - `L_kl = (KL_map * M).sum() / (M.sum() + 1e-6)`

#### Modo SOFT (opcional, pode ser melhor depois)
- Peso contínuo:
  - `M = (pT ** kl_mask_power)`  (power=1.0 inicialmente)
- Loss normalizada:
  - `L_kl = (KL_map * M).sum() / (M.sum() + 1e-6)`

#### Modo NONE
- Volta ao KL denso atual:
  - `L_kl = KL_map.mean()`

---

### 3) Instrumentação / logs (para validar que funcionou)
Além do que já existe (`mean`, `nonfinite_steps`, `clamp_hits`, `pT/pS range`), adicionar:

- `kl_mask_active_ratio`:
  - `active = (M > 0).float().mean().item()` (hard)  
  - ou `active = (M.mean()).item()` (soft)
- Log por epoch:
  - `KLmask: mode=hard tau=0.01 active=...`

**Critério esperado após a mudança:**
- `active_ratio` deve ser pequeno (ex.: 0.1% a 5%, depende do tau)
- `clamp_hits` deve cair bastante (não precisa zerar, mas cair muito)
- `L_kl_mean_epoch` deve continuar finito e estável

---

### 4) Guard de estabilidade (manter)
Manter:
- cálculo em FP32 fora do autocast
- clamp com `eps=1e-4`
- `if not isfinite(L_kl): L_kl=0; kl_nonfinite_steps++`

---

## Checklist de validação (o agente deve responder após patch)
1) `L_kl` agora usa `L_kl = (KL_map*M).sum()/(M.sum()+1e-6)` no modo hard.
2) Confirmar que `M` vem de `pT` (teacher) e **não** do student.
3) Log do epoch 0 deve incluir:
   - `active_ratio` e `tau`
4) Rodar 1 epoch com `--lambda_kl 0.25` e confirmar:
   - `nonfinite_steps=0`
   - `active_ratio` > 0 (não pode ser zero o tempo todo)
   - `clamp_hits` significativamente menor que antes

---

## Sugestão de parâmetros iniciais para experimento
Começar com:
- `--lambda_kl 0.25`
- `--kl_mask_mode hard`
- `--kl_tau 0.01`

Se `active_ratio` ficar quase zero, diminuir tau:
- `--kl_tau 0.005` ou `0.001`

Se `active_ratio` ficar grande demais, aumentar tau:
- `--kl_tau 0.05`
