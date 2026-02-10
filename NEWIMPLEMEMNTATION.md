Você é um agente de implementação (5.2-codex-max). Modifique o ConfMix (YOLOv5) no arquivo `uda_train.py` (branch ldistill) para implementar uma nova arquitetura de treinamento UDA com teacher-student distillation cross-domain, conforme especificação abaixo. NÃO altere a lógica principal de mix/target do ConfMix, exceto pela substituição explícita indicada no merge do termo de consistência. Se necessário pode verificar também os arquivos `loss.py`, `dataloaders.py` ou qualquer arquivo da pasta `utils`.

========================================================
RESUMO DO ESTADO ATUAL (branch ldistill)
========================================================
- Existe suporte a teacher carregado via `--teacher_weights`, congelado (eval, no grad).
- Existe flag `--use_distill`, `--lambda_distill`, `--distill_conf_thres`.
- Implementação atual faz "hard distill": teacher gera pseudo-label boxes em S' e student treina ComputeLoss com pseudo-labels.
- ConfMix original ainda calcula L_cons no mixed image e faz merge de previsões source+target.

========================================================
OBJETIVO DA NOVA MODIFICAÇÃO
========================================================
Trocar o hard distill em S' por uma abordagem com:
(1) Par supervisionado por amostra: (x_s, x'_s) onde:
    - x_s = imagem source original (Sim10K)
    - x'_s = imagem translated (CycleGAN(S->T)) alinhada 1-para-1 com x_s
    - y_s = GT da imagem source (mesmas labels valem para x'_s, pois geometria é preservada)
(2) Teacher recebe x_s e gera predições y~_s (SEM gradiente).
(3) Student recebe x'_s e gera predições y~'_s.
(4) Ramo supervisionado:
    - L_det: COMPUTELOSS padrão do YOLOv5 entre y~'_s (student em x'_s) e y_s (GT).
      -> NÃO calcular L_det usando x_s.
    - Novo termo L_kl: KL SOMENTE SOBRE OBJECTNESS entre teacher(x_s) e student(x'_s).
      -> Não usar pseudo-boxes, não usar NMS, não usar distill_conf_thres para gerar labels.
(5) Termo de consistência (L_cons) e mixing (target) ficam “praticamente intocados”:
    - Continuar calculando y~_m = student(x_mixed) e y~_t = student(x_t) como no ConfMix.
    - A ÚNICA mudança no L_cons: o merge "y~_{s,t}" deve usar o branch do student em x'_s:
          Antes (ConfMix / ou ldistill): y~_{s,t} = merge(y~_s, y~_t)
          Agora:                          y~_{s,t} = merge(y~'_s, y~_t)
      Ou seja, substitua o termo source usado no merge (para L_cons) por y~'_s.

========================================================
DETALHAMENTO DO L_kl (OBJECTNESS ONLY)
========================================================
Precisamos de um termo KL entre mapas de objectness do teacher e do student.
- Saídas do YOLOv5: normalmente `pred` é uma lista por escala, cada tensor com shape [bs, na, ny, nx, no].
  Onde no = 5 + nc (obj logit em index 4 antes do sigmoid, dependendo do código).
- Implementar L_kl assim:
  1) Faça forward no teacher com x_s e obtenha `pred_T` (mesma estrutura do student).
  2) Faça forward no student com x'_s e obtenha `pred_Sp`.
  3) Extraia SOMENTE o logit (ou prob) de objectness de cada escala:
       obj_logits = p[..., 4]  (confirmar índice no código do YOLOv5 usado)
     Use a mesma extração para teacher e student.
  4) Converta para distribuição com sigmoid:
       pT = sigmoid(obj_logits_T)
       pS = sigmoid(obj_logits_S)
     (pT deve ser detach/stopgrad)
  5) Para KL binário, use forma estável:
       KL(pT || pS) = pT * log((pT+eps)/(pS+eps)) + (1-pT)*log((1-pT+eps)/(1-pS+eps))
     Some/mean sobre todas as escalas, anchors e células e batch.
     Use eps=1e-6 ou similar para estabilidade.
  6) L_kl deve ser escalado por um peso `lambda_kl` (pode reutilizar --lambda_distill para isso, ou criar nova flag --lambda_kl; preferível criar nova para não confundir, mas se quiser manter compatibilidade, reaproveite e renomeie internamente).
- Importante: NÃO aplicar KL em bbox nem em cls.

========================================================
LOSS TOTAL
========================================================
Manter a forma geral:
  L_total = L_det + L_cons + lambda_kl * L_kl
onde:
- L_det = ComputeLoss(student_pred_on_xprime, y_s)
- L_cons = como antes, MAS usando merge(y~'_s, y~_t)
- L_kl = KL_binário(objectness_teacher(xs) || objectness_student(xprime))

========================================================
DADOS / DATALOADER
========================================================
- É necessário obter x_s e x'_s pareados no mesmo step.
- Preferência: adaptar o loader/source dataset para retornar (img_s, img_sprime, targets)
  onde img_sprime é a versão translated do mesmo índice.
- Assuma que existe uma raiz/estrutura para translated semelhante à source; se já existe no código um caminho para S', use-o.
- Garanta que augmentations geométricas (resize/flip) sejam aplicadas de forma CONSISTENTE ao par (x_s, x'_s), ou então desative aug geométrica no par. (Esse ponto é crítico: KL e L_det assumem alinhamento espacial.)
- Se no código atual as augs são randômicas por imagem, você deve implementar “paired transforms”: mesma seed/params para ambos.

========================================================
FLAGS / INTERFACE
========================================================
- Manter `--teacher_weights` e `--use_distill`.
- Remover/ignorar `--distill_conf_thres` (não será usado).
- Adicionar uma flag nova (se possível):
    --lambda_kl (default 0.25)
  e manter `--lambda_distill` apenas para backward-compatibility (ou mapear lambda_distill -> lambda_kl).
- Logar durante treino:
    - valor de L_det, L_cons, L_kl (separados)
    - valor efetivo de lambda_kl

========================================================
CHECKLIST DE IMPLEMENTAÇÃO
========================================================
1) Remover geração de pseudo-boxes do teacher em S'.
2) Inserir forward teacher(xs) e student(x'_s) no ramo supervisionado.
3) Calcular L_det usando student(x'_s) com targets GT y_s.
4) Calcular L_kl (objectness only) entre teacher(xs) e student(x'_s).
5) No cálculo do L_cons, substituir o componente source do merge para usar y~'_s (student(x'_s)).
6) Garantir paired augmentation para xs e x'_s.
7) Manter teacher congelado e sem gradiente (detach em pT).
8) Garantir que o código rode com batch=2, img=600, epochs=50 como antes.

========================================================
SAÍDA ESPERADA
========================================================
Um commit funcional em `uda_train.py` (e onde mais precisar: dataset/loader) que implementa exatamente esse fluxo, sem quebrar o ConfMix para o restante do pipeline. Não altere a parte de pseudo-label do target e a estratégia de mixing/confidence do ConfMix, exceto a substituição do merge indicada.
