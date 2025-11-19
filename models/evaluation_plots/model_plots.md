# ��� Visualization Assets - Quick Reference

**Date:** November 19, 2025  
**Status:** Production Ready  
**Total:** 20 plots across 5 equipment types

---

## ��� Quick Access

### Turbine (5 plots)

- ✅ `models/evaluation_plots/turbine/rul_timeseries_plot.png`
- ✅ `models/evaluation_plots/turbine/model_comparison.png`
- ✅ `models/evaluation_plots/turbine/residuals_plot.png`
- ✅ `models/evaluation_plots/turbine/feature_importance.png`
- ✅ `models/evaluation_plots/turbine/optuna_optimization_history.png`

### Compressor (4 plots + 1 SHAP)

- ✅ `models/evaluation_plots/compressor/efficiency_scatter.png`
- ✅ `models/evaluation_plots/compressor/rul_prediction.png`
- ✅ `models/evaluation_plots/compressor/xgboost_comparison.png`
- ✅ `models/evaluation_plots/compressor/anomaly_confusion_matrix.png`
- ✅ `models/metrics/compressor/compressor_shap_combined.png`

### Pipeline (3 plots + 1 SHAP)

- ✅ `models/evaluation_plots/pipeline/confusion_matrix.png`
- ✅ `models/evaluation_plots/pipeline/risk_distribution.png`
- ✅ `models/evaluation_plots/pipeline/feature_correlation.png`
- ✅ `models/metrics/pipeline/pipeline_shap_importance.png`

### Bearing (4 plots)

- ✅ `models/evaluation_plots/bearing/anomaly_score_distribution.png`
- ✅ `models/evaluation_plots/bearing/fft_spectrum_comparison.png`
- ✅ `models/evaluation_plots/bearing/anomaly_rate_comparison.png`
- ✅ `models/evaluation_plots/bearing/feature_importance.png`

### Pump (4 plots)

- ✅ `models/evaluation_plots/pump/health_score_distribution.png`
- ✅ `models/evaluation_plots/pump/efficiency_degradation.png`
- ✅ `models/evaluation_plots/pump/temp_vibration_correlation.png`
- ✅ `models/evaluation_plots/pump/anomaly_comparison.png`

---

## ��� Documentation

- **Full details:** `models/evaluation_plots/README.md`
- **Model selection:** `MODEL_SELECTION_RESULTS.md` (with embedded images)
- **Web data:** `MVP/Web_tinh/data/evaluation_plots.json`

---

## ��� Usage

### In Markdown:

```markdown
![Plot Name](models/evaluation_plots/turbine/rul_timeseries_plot.png)
```

### In HTML:

```html
<img
  src="models/evaluation_plots/turbine/rul_timeseries_plot.png"
  alt="Turbine RUL"
/>
```

### In JavaScript:

```javascript
const plots = await fetch("MVP/Web_tinh/data/evaluation_plots.json").then((r) =>
  r.json()
);
console.log(plots.turbine.visualizations);
```
