# ConfMix + KL(Objectness) — Checklist de validação (perguntas para o agente)

## 1) Flags e defaults
- [x] Flags novas: `--lambda_kl` (default 0.25) e `--lambda_distill` deprecated mapeando para `lambda_kl`; `--distill_conf_thres` marcado como deprecated/unused. Ver [uda_train.py](uda_train.py#L764-L767).
- [x] Default de `lambda_kl` = 0.25. Ver [uda_train.py](uda_train.py#L764).
- [x] Se nao passar `--lambda_kl`, o KL usa o default 0.25 (ou `--lambda_distill` se informado). Ver [uda_train.py](uda_train.py#L338-L339).
- [x] `--distill_conf_thres` ainda existe, mas esta inativo/ignorado. Ver [uda_train.py](uda_train.py#L767).

## 2) Pareamento S ↔ S′
- [x] Como o código garante o pareamento 1–1 entre `x_s` e `x'_s`?
  - é por **mesmo índice** do Dataset?
  - ou por **nome de arquivo** (ex.: `000123.png`)?
  - Resposta: o pareamento e por **stem do nome do arquivo** (basename sem extensao) e alinhado no init do dataset. Ver [utils/dataloaders.py](utils/dataloaders.py#L1068-L1087).
- [x] O DataLoader faz shuffle no nivel do **par** (dataset unico `LoadImagesAndLabelsPair`). Nao ha dois loaders sincronizados para S e S'. Ver [utils/dataloaders.py](utils/dataloaders.py#L237-L278).
- [x] Se faltar par correspondente, ocorre erro (exception) listando alguns nomes faltantes. Ver [utils/dataloaders.py](utils/dataloaders.py#L1078-L1087).

## 3) Paired augmentation (crítico)
- [x] Quais augmentations **geometricas** estao ativas no par? Letterbox (resize/pad), random_perspective e flips; mosaic/mixup estao desligados para o par. Ver [utils/dataloaders.py](utils/dataloaders.py#L1043-L1046) e [utils/dataloaders.py](utils/dataloaders.py#L1124-L1168).
- [x] Como o código garante que os parametros geometricos sao **identicos** em `x_s` e `x'_s`?
  - mesma seed?
  - mesma chamada de transformação aplicada em ambos?
  - transform function que recebe e retorna o par?
  - Resposta: o `random_perspective` roda duas vezes com o mesmo estado de RNG salvo/restaurado; assim aplica exatamente os mesmos parametros geometricos aos dois. Ver [utils/dataloaders.py](utils/dataloaders.py#L1129-L1145).
- [x] Augmentations de **cor** (HSV/brightness/contrast) estao:
  - aplicadas apenas em `x'_s`?
  - ou aplicadas em ambos?
  - ou independentes (permitido)?
  - Resposta: HSV e aplicado apenas em `x'_s`. Ver [utils/dataloaders.py](utils/dataloaders.py#L1153-L1156).

## 4) Definição exata do L_kl (objectness only)
- [x] Em qual tensor do YOLOv5 esta sendo extraido objectness?
  - confirmar o índice (geralmente `[...,4]` antes do sigmoid)
  - Resposta: usa `[...,4]` como logit de objectness. Ver [uda_train.py](uda_train.py#L559-L562).
- [x] O KL e calculado sobre:
  - logits (antes do sigmoid) ou probabilidades (após sigmoid)?
  - Resposta: usa probabilidades apos sigmoid. Ver [uda_train.py](uda_train.py#L559-L562).
- [x] A formula usada e KL binario completo?
  - `pT*log(pT/pS) + (1-pT)*log((1-pT)/(1-pS))`
  - Resposta: sim, exatamente essa formula. Ver [uda_train.py](uda_train.py#L561-L563).
- [x] `pT` (teacher) esta com `.detach()` / `no_grad()` garantido? Sim, `teacher` roda em `torch.no_grad()` e `p_t` e `detach()`. Ver [uda_train.py](uda_train.py#L552-L562).
- [x] Reducao do L_kl: media por escala (mean nos mapas e media entre escalas). Ver [uda_train.py](uda_train.py#L558-L565).

## 5) Mudança no L_cons (swap no merge)
- [x] Confirma que `L_cons` mantem:
  - mesmo gamma/weight schedule
  - mesma lógica de filtro do ConfMix no target
  - Resposta: gamma e calculado igual (mesma logica) e targets_confmix nao foram alterados. Ver [uda_train.py](uda_train.py#L511-L523) e [uda_train.py](uda_train.py#L548-L549).
- [x] Confirma que a unica alteracao foi:
  - `y_{s,t} = merge(y'_s, y_t)` no lugar de `merge(y_s, y_t)`
  - Resposta: em modo distill, `out_s` vem de `pseudo_sp` (student em `x'_s`) e e usado no merge. Ver [uda_train.py](uda_train.py#L422-L432) e [uda_train.py](uda_train.py#L511-L514).

## 6) Custo e device
- [x] Teacher esta no **mesmo device** do student (GPU) via `attempt_load(..., device)`. Ver [uda_train.py](uda_train.py#L224-L231).
- [x] Teacher roda dentro do mesmo bloco de autocast; nao ha conversao explicita para half (fica com AMP quando habilitado). Ver [uda_train.py](uda_train.py#L405-L407) e [uda_train.py](uda_train.py#L552-L555).
- [x] O forward do teacher roda dentro de `torch.no_grad()`. Ver [uda_train.py](uda_train.py#L552-L555).

## 7) Logs e sanity checks
- [x] O treino loga separadamente `L_det`, `L_cons`, `L_kl` no tqdm e no log por epoca. Ver [uda_train.py](uda_train.py#L621-L636).
- [x] Nao existe print/assert de shapes para `pred_T` vs `pred_S`. (nao implementado).
- [x] Nao existe verificacao opcional de geometria apos aug. (nao implementado).
