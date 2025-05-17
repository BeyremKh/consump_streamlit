# Energy Consumption Analysis App

A Streamlit application for analyzing and visualizing energy consumption patterns, PV production, and battery storage performance.

## Features

- Interactive visualization of energy consumption and production
- PV system simulation with configurable parameters
- Battery storage modeling
- Self-consumption and self-sufficiency metrics
- Custom error simulation for model comparison

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/BeyremKh/consump_streamlit.git
   cd consump_streamlit
   ```

2. **Set up the environment**
   Using `uv` (recommended for speed):
   ```bash
   pip install uv
   uv venv
   .venv\Scripts\activate  # On Windows
   # or
   # source .venv/bin/activate  # On Unix/Mac
   
   uv pip install -e .
   ```

   Or using pip:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app locally:
```bash
streamlit run self_consumption/streamlit_app.py
```

## Project Structure

```
.
├── .venv/                  # Virtual environment (ignored by git)
├── .gitignore              # Git ignore file
├── pyproject.toml          # Project configuration
├── README.md               # This file
└── self_consumption/       # Main package
    ├── __init__.py         # Package initialization
    ├── streamlit_app.py    # Streamlit application
    └── self_consumption_analysis.py  # Core analysis logic
```

## Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" and connect your repository
4. Set the main file path to `self_consumption/streamlit_app.py`
5. Click "Deploy!"

## Dependencies

- Python 3.8+
- Streamlit
- NumPy
- Pandas
- Plotly
- Matplotlib

## License

MIT License - see [LICENSE](LICENSE) for details.
