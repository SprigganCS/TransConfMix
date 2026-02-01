> **PROMPT PARA IMPLEMENTAÇÃO**

Você deve modificar o código do ConfMix (arquivo `uda_train.py`) que utiliza YOLOv5 como detector, implementando um esquema de *teacher–student distillation* **somente na loss de detecção do ramo supervisionado**, sem alterar a loss de consistência (`L_cons`) nem o mecanismo de confidence-based region mixing.

### **Contexto do código**

* O pipeline atual do ConfMix possui:

  * imagens source (S) com GT → supervisionadas por `L_det`
  * imagens target (T) sem GT → usadas para pseudo-label + mixing
  * loss total: `L = L_det + L_cons`
* YOLOv5 é usado com sua loss padrão (`ComputeLoss`)
* O arquivo principal de treino é `uda_train.py`

### **Objetivo da modificação**

Adicionar suporte a um terceiro tipo de dado:

* **Translated source (S')** (imagens CycleGAN(S→T)), que possuem GT herdado de S, **mas NÃO devem usar esses GTs**.

Para imagens (S'), a loss de detecção deve ser substituída por uma **loss de destilação baseada em pseudo-labels gerados por um professor**.

---

## **Requisitos funcionais**

### 1. Professor (Teacher Model)

* Carregar um checkpoint YOLOv5 treinado **apenas em S**
* O teacher:

  * deve estar em `eval()`
  * deve estar com `requires_grad = False`
  * nunca deve receber gradientes

### 2. Geração de pseudo-labels (Opção A — hard distillation)

Para cada batch de imagens (x_{S'}):

1. Executar forward do **teacher** em `x_{S'}`
2. Aplicar:

   * score threshold configurável (ex: 0.5)
   * NMS padrão do YOLOv5
3. Converter as detecções resultantes em pseudo-labels no formato YOLO:

   ```
   class_id x_center y_center width height
   ```

   (normalizados pela resolução da imagem)
4. Se nenhuma detecção sobreviver ao threshold:

   * **não computar `L_det` para essa imagem**

### 3. Loss de detecção modificada

* Para batches de **S**:

  * usar `ComputeLoss` normalmente com GT real
* Para batches de **S'**:

  * usar `ComputeLoss` com pseudo-labels do teacher
  * **não usar os GT originais de S'**
* Introduzir um peso escalar `lambda_distill` (ex: 0.5):

  ```
  L_det = lambda_distill * L_det_distill
  ```

### 4. Integração com o ConfMix

* NÃO modificar:

  * confidence-based region mixing
  * seleção de regiões no target
  * cálculo de `L_cons`
* O fluxo de mixing deve continuar usando:

  * imagens source reais (S), não S'
* As imagens (S') entram **apenas no ramo supervisionado superior**, via distilação

### 5. Controle por flags

Adicionar flags/configs:

* `--use_distill`
* `--lambda_distill`
* `--teacher_weights`
* `--distill_conf_thres`

---

## **Resumo esperado do fluxo**

1. Batch de S → `L_det` normal
2. Batch de S' → `teacher → pseudo-label → L_det_distill`
3. Batch de T → mixing + `L_cons`
4. Backprop:

   ```
   L_total = L_det(S) + lambda_distill * L_det(S') + L_cons
   ```

---

## **Observações importantes**

* Reutilizar ao máximo funções já existentes no YOLOv5 (NMS, loss, label parsing).
* Garantir que pseudo-labels não gerem crash quando vazios.
* Garantir que teacher e student usem o mesmo input preprocessing.

## **Informações complementares**
Atualmente eu uso para o UDA o comando:
python uda_train.py  --name {NAME}  --batch 2  --img 600  --epochs 50  --data data/Sim10K2Cityscapes.yaml  --weights {model.pt}

mas isso pode ser alterado. O importante é que seja pensado também se o .yaml deve ser alterado. Atualmente o usado é o do endereço:
/home/andre/ConfMix/data/Sim10K2Cityscapes.yaml
