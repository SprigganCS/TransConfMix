# Revisão do texto acadêmico — ConfMix-Distill

## 1. "preserves the original data flow of ConfMix"

**Impreciso.** A estrutura/lógica do ConfMix é preservada (quadrantes, gamma, mixing, consistency), mas o data flow muda:

- `L_det` usa `pred_sp` (student em x'_s) com GT, não `pred_s` (student em x_s)
- O mixing usa `mix(x'_s, x_t)`, não `mix(x_s, x_t)` (linha 533)
- Os pseudo-labels source no merge do L_cons vêm de `NMS(student(x'_s))`, não `NMS(student(x_s))`

Na prática, x_s é usado **apenas** como input do teacher. Todo o resto (L_det, mixing, merge do L_cons) opera sobre x'_s.

**Sugestão:** trocar "preserves the original data flow" por "preserves the original adaptation structure/mechanism" ou "retains the confidence-based mixing and consistency framework".

---

## 2. "The teacher is defined by the detector obtained after the first stage of ConfMix, while the student is initialized from this same model"

**Correto como protocolo experimental**, mas não é enforced no código. O teacher vem de `--teacher_weights` e o student de `--weights` — podem ser checkpoints diferentes. Se é assim que os experimentos são rodados, OK como descrição do protocolo, mas vale ter consciência de que o código não garante isso automaticamente.

Mesma ressalva vale para o caption da figura: "student model F(θ) initialized from the teacher".

---

## 3. γ controla a contribuição do L_cons

**Correto**, mas pode dar a entender que γ é um hiperparâmetro fixo (como λ_KL). Na implementação, γ é **dinâmico por batch** — calculado como a fração de pseudo-labels com confiança > 0.5:

```python
gamma = (targets_confmix[:,6] > c_gamma_thres).sum() / (targets_confmix[:,6]).nelement()
```

Se o nível de detalhe do texto exige, vale esclarecer isso.

---

## 4. Omissão: mixing usa x'_s em vez de x_s

O texto não menciona que, no ConfMix-Distill, a imagem mixed é construída com x'_s (translated) em vez de x_s (original). Isso é uma mudança real na implementação (linha 533: `mix_source = imgs_sp_cpu if opt.use_distill else imgs_s`).

Deveria ser mencionado explicitamente que x'_s substitui x_s em:
- L_det (supervisão com GT)
- Mixing (construção de x_M)
- Merge do L_cons (pseudo-labels source)

E que x_s é usado **exclusivamente** como input do teacher para o L_KL.
