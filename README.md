# Behavioral Monitoring System

A Python-based system for analyzing behavioral data from Virmen experiments. This project provides tools for loading, processing, and visualizing behavioral data from MATLAB files, with support for various experimental paradigms.

## Features

- **Data Loading**: Load and process behavioral data from MATLAB (.mat) files
- **Data Analysis**: Calculate performance metrics, psychometric curves, and learning curves
- **Visualization**: Generate publication-ready plots for behavioral data
- **Statistical Modeling**: Implement GLM models for behavioral analysis
- **Multi-session Analysis**: Process data across multiple experimental sessions
- **Optogenetic Analysis**: Specialized analysis for optogenetic experiments

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda (recommended) or pip

### Option 1: Using Conda (Recommended)

```bash
git clone https://github.com/yourusername/behavioral-monitoring.git
cd behavioral-monitoring
conda env create -f environment.yml
conda activate behavioral-monitoring
```

### Option 2: Using pip

```bash
git clone https://github.com/yourusername/behavioral-monitoring.git
cd behavioral-monitoring
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

1. **Load the main module**:
```python
from daily_behavioral_monitoring import *
load_data = load_behavioral_data()
plot_data = plot_behavioral_data()
```

2. **Analyze a single session**:
```python
mouse_ids = ['CBJB-2-1L']
date = '250325'  # Format: YYMMDD
base_directory = '/path/to/your/behavior/data/'
behavior_data = load_data.get_behavior_data_single_session_all_animals(mouse_ids, date, base_directory)
plot_data.plot_behavior_single_session(mouse_ids, date)
```

3. **Run analysis across multiple sessions**:
```python
plot_data.plot_behavior_across_sessions(mouse_ids, base_directory)
```

## Data Format

The system expects behavioral data in MATLAB (.mat) format with the following naming convention:
- `{mouse_id}_{date}_Cell.mat` for cell data
- `{mouse_id}_{date}.mat` for session data

Where:
- `mouse_id`: Mouse identifier (e.g., 'CBJB-2-1L')
- `date`: Date in YYMMDD format (e.g., '250325' for March 25, 2025)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 