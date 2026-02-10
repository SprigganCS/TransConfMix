# Patch requerido: usar S′ no mixing (x_mixed = mix(xs′, xt))

## Contexto (o que está errado hoje)
Atualmente, o `x_mixed` está sendo construído como `mix(imgs_s, imgs_t)` (xs + xt).
Para a arquitetura definida, o correto é:
- xs (original) é usado **somente** para teacher(xs) → pred_s_teacher (para L_kl)
- todo o restante do student (L_det e L_cons) deve ser baseado em **xs′ (translated)**, incluindo o mix:
  ✅ `x_mixed = mix(xs′, xt)`

## Objetivo do patch
Trocar a base do mix de `imgs_s` para `imgs_sp`:
- de: `imgs_confmix = confmix(imgs_s, imgs_t, ...)`
- para: `imgs_confmix = confmix(imgs_sp, imgs_t, ...)`

## Requisitos de consistência (não quebrar o ConfMix)
1) A máscara/região selecionada no target (pela confiança) continua vindo de `pred_t` (student em xt) como antes.
2) O merge/targets do ConfMix (`targets_confmix` ou equivalente) deve continuar correto para comparar com `pred_m = student(imgs_confmix)`.
3) **Qualquer ajuste geométrico** aplicado ao par (xs, xs′) já está sincronizado; portanto, usar xs′ no mix é consistente.

## O que alterar (passo a passo)
1) Encontre o trecho onde o batch é dividido:
   - `imgs_s` (xs original)
   - `imgs_sp` (xs′ translated)
   - `imgs_t` (xt target)

2) Localize a linha atual do mixing (você citou que está em `uda_train.py` ~L526-L528), algo como:
   - `imgs_confmix = ... (imgs_s, imgs_t, ...)`

3) Troque para:
   - `imgs_confmix = ... (imgs_sp, imgs_t, ...)`

4) Garanta que `pred_m = model(imgs_confmix, ...)` permanece igual.

5) Verifique se existe qualquer lugar que assuma implicitamente que o "source image" do mix é `imgs_s`.
   - Ex: se a função de mixing usa shapes/ids para construir máscara, confirme que funciona com `imgs_sp`.

## Sanity checks obrigatórios após o patch
- [ ] Log/print temporário (apenas 1 época ou 10 iters) confirmando:
  - `imgs_confmix.shape == imgs_sp.shape == imgs_s.shape` (mesmas dims)
  - `imgs_confmix` foi construído a partir de `imgs_sp` (pode logar um boolean ou comentário)
- [ ] Confirmar que o treino roda sem crash e que `L_cons` continua computando.
- [ ] Confirmar que o `targets_confmix` continua vindo de `merge(pred_sp, pred_t)` (isso já está correto).

## Resultado esperado
Após o patch:
- `L_det` continua sendo `ComputeLoss(pred_sp, targets_s)`
- `L_kl` continua entre `teacher(xs)` e `student(xs′)` (obj only)
- `x_mixed` passa a ser `mix(xs′, xt)`
- `L_cons` continua comparando `pred_m` com `merge(pred_sp, pred_t)` sem outras mudanças
