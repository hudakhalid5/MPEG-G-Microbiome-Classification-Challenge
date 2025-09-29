# Carbon Emissions Summary - MPEG-G Microbiome Project

## Carbon Emissions Summary Table

| Component | Duration | Carbon Score (kg CO₂eq) | Carbon Score (g CO₂eq) | Percentage of Total | Impact Level |
|-----------|----------|-------------------------|------------------------|-------------------|--------------|
| **training_data_processing** | 19.3 hours | 0.0758 | 75.8 | 37.0% | 🔴 Very High |
| **test_data_processing** | 5.7 hours | 0.0225 | 22.5 | 11.0% | 🟡 Moderate |
| **feature_engineering** | 7.8 minutes | 0.000509 | 0.509 | 0.2% | 🟢 Low |
| **prediction_generation** | 3.1 minutes | 0.000206 | 0.206 | 0.1% | 🟢 Low |
| **xgboost_training** | 2.7 minutes | 0.000174 | 0.174 | 0.1% | 🟢 Low |
| **zip_extraction** | 2.3 minutes | 0.000152 | 0.152 | 0.1% | 🟢 Low |
| **ml_data_loading** | 50.3 seconds | 0.0000548 | 0.0548 | <0.1% | 🟢 Minimal |


## Summary Statistics
- **Total Carbon Footprint**: 0.205 kg CO₂eq (205 g CO₂eq)
- **Total Runtime**: 27.02 hours (97,276 seconds)
- **Average Emissions Rate**: 1.09 × 10⁻⁶ kg CO₂eq/second
- **Top 2 Components**: Account for 88.7% of total emissions
- **System**: Intel i5-8250U CPU, 16GB RAM, Windows 10

## Carbon Tracking Implementation

This project implements comprehensive carbon footprint tracking using [CodeCarbon](https://codecarbon.io/) to monitor the environmental impact of machine learning workflows.

### Setup and Configuration

#### Installation
```bash
pip install codecarbon>=2.2.0
```

#### Implementation in Code
```python
from codecarbon import EmissionsTracker

# Initialize tracker for each major component
tracker = EmissionsTracker(
    project_name="MPEG-G-microbiome_component",
    experiment_id="microbiome-classification",
    output_dir="./carbon_tracking/",
    output_file="emissions.csv"
)

# Track emissions for each step
tracker.start()
# ... your code here ...
emissions = tracker.stop()
print(f"🌱 Component emissions: {emissions:.6f} kg CO2")
```

### Tracked Components

The carbon tracking covers all major pipeline components:

1. **Data Processing Steps**
   - ZIP extraction
   - MGB → K-mer conversion (Docker-based)
   - Matrix creation

2. **Machine Learning Pipeline**
   - Feature engineering (CLR transformation)
   - Model training (XGBoost with cross-validation)
   - Prediction generation

3. **Visualization Generation**
   - Statistical plots
   - Model performance analysis
   - Dimensionality reduction visualizations

### Environmental Impact Analysis

#### Carbon Intensity Factors
- **Location**: France (Pays de la Loire region)
- **Grid Carbon Intensity**: Based on French electricity mix
- **PUE (Power Usage Effectiveness)**: 1.0 (local machine)

#### Hardware Specifications
- **CPU**: Intel(R) Core(TM) i5-8250U @ 1.60GHz (8 cores)
- **Memory**: 15.88 GB RAM
- **GPU**: None (CPU-only processing)
- **Operating System**: Windows 10

### Optimization Opportunities

Based on the carbon footprint analysis:

1. **High-Impact Areas for Optimization**:
   - **MGB Processing** (88.7% of emissions): Consider cloud-based processing with renewable energy
   - **Batch Optimization**: Larger batch sizes could reduce per-sample overhead

2. **Low-Impact Operations**:
   - Machine learning training and inference are relatively efficient
   - Visualization generation has minimal impact

### Carbon Footprint Context

**205.8g CO₂eq is equivalent to:**
- Driving ~0.5 miles in an average gasoline car
- Charging a smartphone ~25 times
- Running a 60W light bulb for ~57 hours
- One Google search query multiplied by ~2,000 times

### Sustainable Computing Practices

1. **Efficient Processing**:
   - Use checkpointing to avoid re-running failed jobs
   - Implement batch processing for better resource utilization
   - Cache intermediate results when possible

2. **Infrastructure Choices**:
   - Consider cloud providers with renewable energy commitments
   - Use GPU acceleration where appropriate (can be more efficient per operation)
   - Optimize Docker image sizes and container efficiency

3. **Model Development**:
   - Use cross-validation efficiently to avoid redundant training
   - Implement early stopping to prevent unnecessary computation
   - Consider model complexity vs. environmental cost trade-offs

### Reporting and Monitoring

The emissions data is automatically saved to `carbon_tracking/emissions.csv` with detailed metadata:

- **Timestamp**: Exact start time of each component
- **Duration**: Runtime in seconds
- **Emissions**: CO₂ equivalent in kg
- **Energy Breakdown**: CPU, GPU, RAM energy consumption
- **System Info**: Hardware specs, location, carbon intensity