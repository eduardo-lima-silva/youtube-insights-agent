---
tags: [componente]
---

# Predictor de títulos (`predictor.py`)

Pontua até 3 títulos com o [[LightGBM]] + [[TF-IDF + Ridge]] e explica com [[SHAP]] ([[Agente ML]]).
Saída: views previstas, faixa de 2× (empírica, não intervalo de confiança), fatores principais e efeito do título.
Regra de produto: [[ADR-003 Títulos com trade-offs]]. Nunca dizer "melhor título" → [[Associação vs causalidade]].
