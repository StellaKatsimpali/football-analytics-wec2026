# Warsaw Econometric Challenge 2026: Decoding the Final 15 Minutes ⚽📊

**Team:** The Catharsis Crew (Konstantina Sapranidou, Styliani Katsimpali, Pinelopi Karakike)
**Institution:** University of Ioannina, Greece

## Overview
This repository contains our submission for the **Warsaw Econometric Challenge (WEC) 2026**. The core objective of the competition was to answer the question: *"Will you score a goal?"* by predicting the probability of a player scoring in the final 15 minutes of a football match.

By moving beyond standard physical metrics and incorporating technical composure under pressure, we developed a hybrid analytical framework that successfully bridges predictive Machine Learning and causal Econometrics.

## Repository Structure
* `final.R`: The master analytical pipeline (Data Wrangling, Feature Engineering, ML modeling, SHAP analysis, and Fixed-Effects Econometrics).
* `The_Catharsis_Crew_Warsaw_Econometric_Challenge_2026.pdf`: Our comprehensive research paper detailing the methodology and strategic implications.
* `/data` *(if uploaded)*: The match datasets including player appearances, passes, runs, shots, and behavior under pressure.

## Methodology
1. **Advanced Feature Engineering:** We extracted custom metrics such as the *Intensity Index* and *Turnover Rate Under Pressure* from granular event data.
2. **Machine Learning:** We trained an **XGBoost** classifier, achieving an 8.5% AUC-ROC performance uplift by explicitly incorporating technical data alongside physical exertion.
3. **Interpretability:** We utilized **SHAP** (SHapley Additive exPlanations) values to identify the non-linear impacts of fatigue and technical composure.
4. **Causal Inference:** To validate our findings economically, we applied a **Fixed-Effects Generalized Linear Model (GLM)**, effectively controlling for unobserved heterogeneity such as player position and fixture-specific dynamics.

## Key Findings
* Technical metrics (composure under pressure, passing accuracy) are not just supplementary; they are critical in the high-variance final stages of a match.
* Cognitive composure often compensates for physical decay in late-game scenarios.
* Our hybrid model offers a reliable "Compass" for real-time tactical decisions by coaching staff.

## Tech Stack
* **Language:** R
* **Key Libraries:** `tidyverse`, `xgboost`, `shapviz`, `fixest`, `caret`, `pROC`
