# PowerNext-AI 2026 — Methodology Note

**Team:** Kori Rotti  
**Challenge:** The Black-Box Test Bench Challenge (CPRI & MIT Bengaluru)

---

## 1. Approach Used

Our strategy follows a structured four-stage engineering methodology: **Understand → Analyse → Validate → Automate**.

1. **Understand (Ingestion & Diagnostics):** Standardized input schemas, resolved missing sensor values via Multivariate Imputation by Chained Equations (MICE), and conducted statistical diagnostic tests on auxiliary channels.
2. **Analyse (Physics-Informed Feature Engineering):** Engineered a 27-feature physical contract modeling electrical power deposition ($P = V \times I$), thermal energy proxy ($E = P \times t$), resistive Joule heating ($I^2 t$), non-linear convective cooling interactions ($I^2 \times \text{Ambient}$), spatial temperature gradients across incoming/outgoing terminals ($S_1, S_2, S_3$), and logarithmic saturation interaction terms.
3. **Validate (Two-Task Modeling Strategy):**
   - *Task 01 (Validity Classification):* Implemented a hybrid architecture combining spatial Huber/MAD z-score anomaly features with a `HistGradientBoostingClassifier` (stratified 5-fold CV) and deterministic physical safety rules.
   - *Task 02 (Hot-Spot Regression):* Filtered training data strictly to verified `Valid` records (866 rows) to isolate corrupted probe signatures. Implemented a **5-fold bagged tri-model ensemble** (GradientBoosting, XGBoost, and LightGBM) with inverse-variance weighting and physical thermodynamic boundary enforcement.
4. **Automate (Export & Integration):** Programmatically generated `<TeamName>.csv` predictions, ranked high-risk test runs, and produced compliant `summary.json` diagnostic output.

---

## 2. Parameters Considered Important

Through correlation analysis, feature permutation importance, and electrical domain principles, the parameters were ranked by significance:

- **Load Current ($I$) [Highest Importance]:** Demonstrates the strongest direct relationship with hot-spot rise ($r = +0.87$). Resistive heating scales with current squared ($I^2$), making it the primary thermodynamic driver.
- **Sensor $S_2$ (Outgoing Terminal Rise):** Strongest sensor-to-target correlation ($r = +0.70$), indicating proximity to the primary internal thermal bottleneck.
- **Sensors $S_1$ and $S_3$ (Spatial Differential):** Mean rise, maximum rise, and inter-sensor spread ($\Delta S = \text{Max} - \text{Min}$) capture localized hot-spots versus uniform thermal equilibrium.
- **Interaction Terms ($I^2 \times \text{Ambient}$, $P \times \ln(1+t)$):** Model non-linear convective cooling efficiency changes and thermal saturation over time.
- **Sensor $S_4$ (Excluded as Noise):** Cross-correlation with target ($r = 0.0021$) and lag-1 autocorrelation ($\rho_1 = 0.0300$) proved $S_4$ to be unphysical white noise; it was completely eliminated from feature space to prevent tree-split variance.

---

## 3. Method Used for Detecting Abnormal Data (Task 01)

Abnormal and invalid test runs are identified using a two-tier hybrid screening mechanism:

1. **Spatial Anomaly & ML Classifier:** Spatial z-scores ($\text{MAD}$-normalized deviations of each sensor relative to the operating regime) are computed alongside sensor spreads. A `HistGradientBoostingClassifier` trained with L2 regularization ($\lambda = 1.5$) achieved **0.9926 F1-score** and **100.0% recall (134/134)** on 5-fold stratified cross-validation.
2. **Deterministic Physics Safety Overrides:** Direct physical boundary rules override model output to guarantee safety:
   - Negative electrical inputs ($V < 0$, $I < 0$, $t < 0$).
   - Impossible thermal values ($S < -10^\circ\text{C}$ or $S > 400^\circ\text{C}$).
   - Severe sensor disconnection spread ($\text{Max}(S) - \text{Min}(S) > 200^\circ\text{C}$).

On the 350-instance test set, the system flagged **46 abnormal/invalid records (13.1%)**, matching the historical training baseline (13.4%).

---

## 4. Key Engineering Assumptions Made

1. **Temperature Rise Units:** `Sensor_S1`, `Sensor_S2`, `Sensor_S3`, and `Reference_Parameter` are already expressed as temperature rises above ambient ($\Delta T$ in $^\circ\text{C}$). Ambient temperature was not subtracted from sensor readings a second time.
2. **Ambient Role in Heat Dissipation:** `Ambient_Temperature` acts as an independent boundary condition governing external convection and radiation cooling efficiency.
3. **Invalid Record Mechanism:** Invalid records represent data corruption, probe disconnections, or test setup failures. Excluding them during regression training prevents the model from learning artificial relationships.
4. **Thermodynamic Hot-Spot Constraint:** For any active valid test, the hot-spot temperature rise is physically bounded from below by the maximum observed terminal sensor temperature ($\text{Reference\_Parameter} \ge \max(S_1, S_2, S_3) - 0.5^\circ\text{C}$).

---

## 5. Steps for Automated Digital Twin Implementation

To deploy this solution as an automated Digital Twin for real-time test bench monitoring, the following continuous four-step pipeline is established:


1. **Step 1 — Real-Time Streaming Ingestion:** Continuously ingest SCADA test bench telemetry via OPC-UA/MQTT protocols with automated column mapping and schema type-casting.
2. **Step 2 — Dynamic Noise & Channel Filtering:** Run real-time channel health checks (autocorrelation thresholds to detect sensor drift or white noise) and apply multivariate imputation for transient packet dropouts.
3. **Step 3 — Dual-Model Inference Execution:**
   - *Gatekeeper:* Hybrid anomaly classifier screens incoming packets in $<10\text{ ms}$.
   - *Thermal Twin:* Validated packets feed the 5-fold bagged tri-model ensemble (GBR, XGBoost, LightGBM) to infer internal hot-spot rise in real time with physical boundary enforcement.
4. **Step 4 — Automated Prescriptive Maintenance:** Anomalous or high-temperature events trigger automated work orders mapped to specific hardware interventions:

| Detected Signature | Root Cause | Prescriptive Intervention |
|---|---|---|
| $S_2 \gg S_1, S_3$ | Outgoing terminal contact resistance | Re-torque load clamp to 50 Nm; apply silver-plated conductive compound |
| $S_1 \gg S_2, S_3$ | Incoming bushing contact degradation | Clean contact interface; inspect bushing joint compression |
| High Spread ($\Delta S > 20^\circ\text{C}$) | Busbar current crowding / localized joint fault | Upgrade busbar cross-sectional area; inspect bolted joint torque |
| Uniform High $S_1, S_2, S_3$ | Sustained overload / inadequate airflow | Increase cooling fan speed by 25%; clear ventilation duct restrictions |
| Floating / Zero Sensor | Sensor open-circuit or probe detachment | Replace RTD transducer; inspect DAQ ground isolation |

---

*Prepared by Team Kori Rotti for PowerNext-AI 2026 (CPRI & MIT Bengaluru).*