# Safe and Live LLM Guided UAV Navigation via Certified Corridor Feedback

Official implementation of the paper:

**Safe and Live LLM Guided UAV Navigation via Certified Corridor Feedback**

## Abstract

Large Language Models (LLMs) provide a convenient interface for UAV mission planning by translating natural-language instructions into navigation goals. However, LLMs do not provide formal safety guarantees and may repeatedly generate unsafe or infeasible trajectories.

This repository presents a closed-loop framework that combines:

* LLM-based waypoint planning
* High-Order Control Barrier Function (HOCBF) safety filtering
* Deadlock detection
* Control-to-language feedback
* Certified Corridor Feedback (CCF)

The proposed approach guarantees collision avoidance through HOCBF-based safety filtering while restoring liveness through control-grounded feedback that guides LLM replanning.

## Key Features

* Safe UAV navigation using HOCBF-QP filtering
* Real LLM integration (Llama-3.3-70B via Groq API)
* Deadlock detection and recovery
* Certified Corridor Feedback (CCF)
* Visibility-graph-based corridor generation
* Feedback Conversion Efficiency (FCE) evaluation metric
* Fully reproducible experiments

## Method Overview

The navigation loop operates as follows:

```text
Natural Language Mission
          ↓
      LLM Planner
          ↓
       Waypoint
          ↓
    HOCBF Safety Filter
          ↓
    Deadlock Detection
          ↓
 Certified Corridor Feedback
          ↓
     LLM Replanning
          ↓
     Safe Navigation
```

Unlike conventional safety filters that only reject unsafe actions, the proposed framework converts control-derived geometric information into actionable language feedback that enables the LLM to generate safer and more feasible plans.

## Repository Structure

```text
.
├── src/                  # Core implementation
├── environments/         # Environment generation
├── prompts/              # LLM prompts
├── data/                 # Experimental data
├── results/              # Recorded outputs
├── figures/              # Generated figures
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/safe-live-llm-uav-navigation.git
cd safe-live-llm-uav-navigation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## API Configuration

Set your Groq API key:

```bash
export GROQ_API_KEY="YOUR_API_KEY"
```

Alternatively, edit the configuration file with your preferred OpenAI-compatible endpoint.

## Running Experiments

Run the complete evaluation:

```bash
python run_experiments.py
```

Generate all figures:

```bash
python generate_figures.py
```

## Evaluation Metrics

### Collision-Free Success (CFS)

Percentage of missions completed without violating safety constraints.

### Minimum Clearance

Minimum distance maintained from obstacles throughout execution.

### Feedback Conversion Efficiency (FCE)

Measures the effectiveness of feedback relative to the number of replanning operations:

```text
FCE = (CFS_method − CFS_filter_only) / Average_Replans
```

Higher values indicate more efficient feedback.

## Reproducibility

This repository includes:

* Environment generation seeds
* Experimental configurations
* CSV logs
* Figure generation scripts
* Prompt templates

All reported results can be reproduced directly from the provided files.

## Citation

If you use this repository in your research, please cite:

```bibtex
@software{almamoori2026ccf,
  author = {Hayder Almamoori},
  title = {Safe and Live LLM Guided UAV Navigation via Certified Corridor Feedback},
  year = {2026},
  note = {GitHub Repository},
  url = {https://github.com/YOUR_USERNAME/safe-live-llm-uav-navigation}
}
```



## License

MIT License

## Author

Hayder Almamoori

Ege University

## Contact

For questions, suggestions, or collaborations, please open an issue or submit a pull request.

## Acknowledgments

This work was developed as part of research on safe autonomous UAV navigation using Large Language Models and Control Barrier Functions.
