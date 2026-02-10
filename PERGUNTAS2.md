# Confirmação de implementação — ConfMix + KL(obj) com mix em S′

## A) Caminhos de dados (o mais importante)
1) Confirme explicitamente:
   - `teacher` recebe **xs (original)** e só isso.
   - `student` recebe **xs' (translated)** para `L_det` e para o **mix**.
   - `student` recebe **xt (target)** como no ConfMix.
   - Resposta:
     - teacher recebe `imgs[: imgs_s.shape[0]]` (xs original). Ver [uda_train.py](uda_train.py#L552-L556).
     - student recebe `imgs_sp` (xs') para `pred_sp` e `L_det`. Ver [uda_train.py](uda_train.py#L419-L428) e [uda_train.py](uda_train.py#L536-L545).
     - student recebe `imgs[imgs_s.shape[0]:]` (xt) para `pred_t`. Ver [uda_train.py](uda_train.py#L408-L416).

2) Pergunta direta (sim/não):
   - O `x_mixed` é construído como **mix(xs', xt)** (CORRETO) ou **mix(xs, xt)** (INCORRETO)?
   - Indique o nome exato do tensor usado como base do mix (ex.: `imgs_sp` vs `imgs_s`) e o trecho/linha.
   - Resposta: **INCORRETO**. O `x_mixed` e construido com `imgs_s` e `imgs_t` (mix(xs, xt)), nao `imgs_sp`. Ver [uda_train.py](uda_train.py#L526-L528).
   - Tensor usado: `imgs_s` (source original) e `imgs_t` (target). Ver [uda_train.py](uda_train.py#L526-L528).

## B) Definições de predições usadas em cada loss
3) Confirme as definições abaixo e diga onde no código ocorre:
   - `pred_s_teacher = teacher(xs)` (no_grad, eval, detach)
   - `pred_sp = student(xs')`
   - `pred_t = student(xt)`
   - `pred_m = student(x_mixed)`
   - Resposta:
     - `pred_s_teacher = teacher(xs)` em `torch.no_grad()` e `p_t.detach()`. Ver [uda_train.py](uda_train.py#L552-L562).
     - `pred_sp = student(xs')` em `pred_sp = model(imgs_sp, ...)`. Ver [uda_train.py](uda_train.py#L419-L428).
     - `pred_t = student(xt)` em `pred_t = model(imgs[imgs_s.shape[0]:], ...)`. Ver [uda_train.py](uda_train.py#L408-L416).
     - `pred_m = student(x_mixed)` em `pred_confmix = model(imgs_confmix, ...)`. Ver [uda_train.py](uda_train.py#L530-L538).

4) Confirme o cálculo das losses:
   - `L_det = ComputeLoss(pred_sp, GT_source)`  (GT de xs aplicado em xs')
   - `L_kl = KL_binário(sigmoid(obj(pred_s_teacher)) || sigmoid(obj(pred_sp)))` **ONLY obj**
   - `L_cons` compara `pred_m` com `merge(pred_sp, pred_t)` (swap source preds)
   - Resposta:
     - `L_det` usa `pred_sp` com `targets_s`. Ver [uda_train.py](uda_train.py#L536-L543).
     - `L_kl` usa obj `[...,4]` com sigmoid e KL binario completo. Ver [uda_train.py](uda_train.py#L558-L565).
     - `L_cons` usa `targets_confmix` formados com `out_s` vindo de `pred_sp` e `pred_t` do target. Ver [uda_train.py](uda_train.py#L419-L432) e [uda_train.py](uda_train.py#L511-L514).

## C) KL (objectness only) — detalhes tensor a tensor
5) Confirme:
   - objectness extraído com `[...,4]` em todas as escalas
   - teacher em `torch.no_grad()` e `p_teacher.detach()`
   - KL binário completo com eps (1e-6)
   - redução: mean por mapa e mean entre escalas
   - Resposta: tudo confirmado em [uda_train.py](uda_train.py#L552-L565).

## D) Dataset/augmentations (pareamento e alinhamento espacial)
6) Confirme:
   - xs e xs' são pareados 1–1 por stem/filename
   - shuffle acontece no nível do par (um único dataset/loader)
   - augmentação geométrica é **paired** (mesmos params em xs e xs')
   - HSV é aplicado apenas em xs' (ok)
   - Resposta:
     - Pareamento por stem/filename: [utils/dataloaders.py](utils/dataloaders.py#L1068-L1087).
     - Shuffle no nivel do par: `LoadImagesAndLabelsPair` em um unico loader. Ver [utils/dataloaders.py](utils/dataloaders.py#L237-L278).
     - Geometria paired: `random_perspective` com RNG salvo/restaurado. Ver [utils/dataloaders.py](utils/dataloaders.py#L1129-L1145).
     - HSV apenas em xs': [utils/dataloaders.py](utils/dataloaders.py#L1153-L1156).

## E) Logs/sanity checks mínimos
7) Confirme que logs mostram separadamente:
   - `L_det`, `L_cons`, `L_kl`, e o valor de `lambda_kl`
8) (Opcional) Adicione um assert no modo distill:
   - `assert pred_T[i].shape == pred_Sp[i].shape` para todas as escalas
   - Resposta:
     - Logs separados: tqdm postfix e log por epoca em [uda_train.py](uda_train.py#L621-L636).
     - Assert opcional: **nao implementado**.
