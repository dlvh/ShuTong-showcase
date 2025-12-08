# ShuTong: Trustworthy Step-by-Step Math Problem Solving

> **ShuTong** (书童, literary Chinese: "personal learning assistant") is a multi-agent system for trustworthy mathematical problem-solving that generates detailed, pedagogically valuable step-by-step solutions with automated correctness verification.

## Overview

While contemporary language models demonstrate strong performance in solving mathematical problems, their solutions often lack clarity and pedagogical value. Reasoning models, though powerful, are computationally expensive to deploy at scale and tend to produce succinct final answers that offer limited learning value for students.

**ShuTong addresses these challenges through an agentic design using non-reasoning models.** Rather than relying on a single model's reasoning capabilities, our multi-agent system generates detailed step-by-step solutions that break down complex problems into comprehensible stages. Each solution is enriched with relevant knowledge points and mathematical concepts, enabling students to not only follow the solution but also understand the underlying principles.

### Key Objectives

1. **Trustworthiness**: Automated verification of each solution step for logical and computational correctness
2. **Pedagogical Value**: Detailed step-by-step explanations that mimic office-hour tutoring
3. **Cost Efficiency**: Achieve strong performance using non-reasoning models with tool augmentation
4. **Error Detection**: Identify the first error in multi-step solutions to prevent cascading mistakes

## System Architecture

ShuTong consists of two specialized agents orchestrated via LangGraph:

### Solver Agent
Generates detailed, step-by-step solutions to mathematical problems using intelligence-first base models for creative problem-solving. The solver breaks down complex problems into individual steps with clear explanations.

### Critic Agent
Evaluates each solution step for correctness using a sophisticated 3-node LangGraph workflow. The critic has access to four SymPy-based calculator tools:

1. **`evaluate_numerical`**: Numerical evaluation with variable substitution
2. **`evaluate_symbolic`**: Symbolic algebraic simplification
3. **`verify_calculation`**: Check if an expression equals an expected result
4. **`compare_expressions`**: Test mathematical equivalence of two expressions

The critic implements up to K rounds of tool calling per step to verify calculations before generating its critique, supporting both LaTeX and standard mathematical notation.

### Iterative Refinement Loop
When the critic detects errors in a solution, it provides constructive feedback to the solver, which then generates a refined solution. This process repeats for a configurable number of iterations until the solution passes all correctness checks.

![ShuTong Architecture](report/shutongArchitecture.png)

## Research & Evaluation

ShuTong was developed and evaluated as part of research on trustworthy AI systems for mathematical education. Our evaluation focuses on the critic agent's ability to detect the **first error** in multi-step solutions—a critical capability for educational applications where early error detection prevents cascading mistakes.

### Evaluation Dataset

We evaluated ShuTong using **ProcessBench** ([Qwen/ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench)), a dataset specifically designed for evaluating process-level supervision in mathematical reasoning. The dataset contains multi-step solutions to Math Olympiad problems with ground truth labels indicating error locations.

### Metrics

We evaluated performance along three complementary axes:

1. **Exact Match**: Identifies the exact step where an error first occurs (fine-grained analysis)
2. **Correct Match**: Binary judgment of whether the solution is correct or incorrect (high-level assessment)
3. **Cost Efficiency**: Average cost per correct prediction in USD

### Results Summary

| Model | Exact Match | Correct Match | Cost per Correct Match |
|-------|-------------|---------------|------------------------|
| **GPT-5 mini (Ours)** | **82.1%** | **93.6%** | **$0.0082** |
| GPT-5.1 Reasoning | 68.2% | 92.8% | $0.0072 |
| GPT-5 nano | 42.5% | 71.4% | $0.0011 |
| GPT-4o-mini | 39.6% | 68.7% | $0.0028 |

**Key Finding**: Our agentic design with GPT-5 mini substantially outperforms the GPT-5.1 reasoning model on exact match accuracy (82.1% vs 68.2%) while maintaining similar cost efficiency. This demonstrates that **tool augmentation and agentic design can achieve superior performance compared to reasoning models alone**.

### Evaluation Data Availability

> **Note**: This public repository contains partial evaluation results due to previous git conflicts in the project history. Complete evaluation results in `.json` format are available upon request. Please contact the authors if you need access to the full evaluation data.

## Features

- **Automated Problem Solving**: Generate comprehensive step-by-step solutions to math problems
- **Tool-Augmented Verification**: Critic uses SymPy-based calculator tools to verify mathematical expressions
- **Iterative Refinement**: Solutions automatically refined based on critic feedback (configurable iterations)
- **Knowledge Extraction**: Identifies and tracks mathematical concepts used in solutions
- **Interactive UI**: Streamlit-based web interface for quick experimentation
- **React Frontend**: Modern web interface with real-time streaming updates via Server-Sent Events
- **Comprehensive Tracking**: SQLite-based tracking system records all agent interactions and token usage
- **Evaluation Framework**: Scripts for evaluating performance on ProcessBench dataset

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Node.js 18+ (for React frontend)

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/ShuTong.git
cd ShuTong

# Install dependencies with uv
uv sync

# Create .env file with your API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Option 2: Using pip and venv

```bash
# Clone the repository
git clone https://github.com/yourusername/ShuTong.git
cd ShuTong

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install the package
pip install -e .

# Create .env file with your API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Frontend Setup (Optional)

```bash
cd frontend
npm install
cd ..
```

## Quick Start

### Option 1: Streamlit UI (Simplest)

The easiest way to use ShuTong is through the Streamlit web interface:

```bash
streamlit run app.py
```

This will open a browser window where you can:
1. Enter a math problem
2. Configure solver and critic models
3. Set iteration limits
4. View the solution with step-by-step critiques
5. See extracted knowledge points

### Option 2: React Frontend + Flask API

For the full web application with real-time streaming:

```bash
# Start both Flask API (port 8000) and React frontend (port 3000)
./start_frontend.sh
```

Or manually:

```bash
# Terminal 1: Start Flask API
python api_server.py

# Terminal 2: Start React frontend
cd frontend
npm start
```

Navigate to `http://localhost:3000` to access the web interface.

### Option 3: Python API

```python
from agent_sys import AgentPipeline

# Initialize the pipeline
pipeline = AgentPipeline(
    solver_model="gpt-4o",
    critic_model="gpt-4o",
    max_iterations=3,  # Number of refinement iterations
    tracker_dir="./data/tracker"
)

# Solve a problem
problem = "Solve the equation: $x^2 - 5x + 6 = 0$"
result = pipeline.run(problem)

# Access results
print(pipeline.format_result(result))
print(f"Final solution: {result['final_solution']}")
print(f"Knowledge points: {result['knowledge_points']}")
```

### Option 4: Using Individual Agents

```python
from solver_agent import Solver
from critic_agent import Critic

# Solver Agent
solver = Solver(model_name="gpt-4o", temperature=0.7)
solution = solver.solve("Integrate: $\int x^2 dx$")
steps = solver.get_solution_steps()

# Critic Agent
critic = Critic(model_name="gpt-4o", temperature=0.3)
all_critiques = critic.critique_all_steps(
    math_problem="Integrate: $\int x^2 dx$",
    solution_steps=steps
)
```

## Running Tests

The project includes test scripts for verifying functionality:

```bash
# Test critic agent with calculator tools
python critic_agent/test_critic_with_tools.py

# Test calculator functionality
python critic_agent/test_calculator.py
python critic_agent/test_calculator_integration.py

# Test the full pipeline
python agent_sys/test_pipeline.py
```

## Running Evaluation

To reproduce our evaluation results on the ProcessBench dataset:

```bash
# Run evaluation
python eval/ProcessBench/eval_processbench.py

# Analyze results
python eval/ProcessBench/analyze.py
```

The evaluation script assesses the critic agent's ability to identify the first error in multi-step mathematical solutions.

## Project Structure

```
ShuTong/
├── agent_sys/              # Agent pipeline integration
│   ├── pipeline.py         # Main AgentPipeline class with LangGraph workflow
│   ├── state.py            # PipelineState definition (TypedDict)
│   └── test_pipeline.py    # Pipeline integration tests
├── solver_agent/           # Solution generation agent
│   ├── solver.py           # Solver implementation
│   └── state.py            # SolverState and SolutionStep definitions
├── critic_agent/           # Solution evaluation agent
│   ├── critic.py           # Critic implementation with tool calling
│   ├── calculator.py       # SymPy-based calculator tools (LaTeX support)
│   └── state.py            # CriticState and StepCritique definitions
├── tracker/                # Function call tracking system
│   └── tracker.py          # SQLite-based tracker decorator
├── eval/                   # Evaluation scripts
│   └── ProcessBench/       # ProcessBench dataset evaluation
│       ├── eval_processbench.py   # Main evaluation script
│       └── analyze.py      # Results analysis
├── frontend/               # React web application
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Main pages (Overview, ProblemGen, AgentSolver)
│   │   ├── services/       # API client with SSE support
│   │   └── utils/          # Helper utilities
│   └── package.json        # Frontend dependencies
├── report/                 # Research paper and documentation
│   ├── cs329t-final-paper.tex
│   └── shutongArchitecture.png
├── data/                   # Data storage
│   └── tracker/            # SQLite database for tracking
├── api_server.py           # Flask API backend with SSE streaming
├── app.py                  # Streamlit web interface
├── start_frontend.sh       # Startup script for full-stack app
├── pyproject.toml          # Python dependencies
└── CLAUDE.md               # Developer documentation for AI assistants
```

## Configuration

### Agent Configuration

Both agents can be configured with different models and parameters:

```python
pipeline = AgentPipeline(
    solver_model="gpt-4o",           # Base model for solver
    critic_model="gpt-4o",           # Base model for critic
    solver_temperature=0.7,          # Higher = more creative
    critic_temperature=0.3,          # Lower = more consistent
    max_iterations=3,                # Number of refinement iterations
    tracker_dir="./data/tracker"     # Directory for tracking DB
)
```

### Critic Tool Iteration Limits

The critic agent limits tool-calling iterations to prevent infinite loops:

```python
critic = Critic(
    model_name="gpt-4o",
    temperature=0.3,
    max_tool_iterations=3  # Default: 3 rounds of tool calls per step
)
```

## How It Works

1. **Problem Input**: User provides a mathematical problem
2. **Initial Solution**: Solver agent generates a step-by-step solution
3. **Step Extraction**: Solution is parsed into individual steps
4. **Critique Phase**: Critic evaluates each step for:
   - Logical correctness
   - Computational accuracy (using calculator tools)
   - Knowledge points used
5. **Refinement Decision**: If errors found AND iterations remain:
   - Feedback is provided to solver
   - Solver generates refined solution
   - Process repeats from step 3
6. **Output**: Final solution with all critiques and knowledge points

## Calculator Tools

The critic agent has access to SymPy-based calculator tools that support:

- **Numerical evaluation**: `2*5 + 3`, `sin(pi/2)`, `sqrt(16)`
- **Symbolic evaluation**: Algebraic simplification and manipulation
- **Expression verification**: Check if two expressions are mathematically equivalent
- **LaTeX support**: Parse and evaluate LaTeX mathematical expressions (e.g., `\frac{1}{2}`, `\int x^2 dx`)

The calculator tools are designed to handle the diverse formatting styles found in ProcessBench, including mixed LaTeX notation, natural language, and computational results.

## API Endpoints

The Flask API server (`api_server.py`) provides the following endpoints:

- `GET /api/health` - Health check
- `GET /api/analysis` - Model performance metrics from tracker database
- `POST /api/generate-problem` - Generate math problems using GPT-4o
- `POST /api/run-agent` - Run pipeline and return full result
- `POST /api/run-agent-stream` - Stream pipeline progress via Server-Sent Events (SSE)
- `GET /api/runs` - Get run history from tracker
- `GET /api/runs/:id` - Get specific run details

## Technology Stack

**Backend (Python)**
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent workflow orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM integration
- [SymPy](https://www.sympy.org/) - Symbolic mathematics
- [Flask](https://flask.palletsprojects.com/) - API backend
- [Streamlit](https://streamlit.io/) - Quick web interface

**Frontend (React)**
- React 18 with React Router
- Axios for HTTP requests
- Server-Sent Events (EventSource) for real-time updates
- KaTeX for LaTeX rendering
- Recharts for data visualization
- TailwindCSS for styling

## Research Paper

For detailed information about the system architecture, evaluation methodology, and results, please refer to our research paper in the `report/` directory:

- **Paper**: `report/cs329t-final-paper.tex`
- **Architecture Diagram**: `report/shutongArchitecture.png`

### Key Findings from Research

1. **Agentic design with tool augmentation** enables non-reasoning models to outperform reasoning models on fine-grained error localization tasks
2. **GPT-5 mini achieved 82.1% exact match accuracy**, nearly double that of GPT-4o-mini (39.6%) and GPT-5 nano (42.5%)
3. **Cost efficiency**: Our approach achieves strong performance at $0.0082 per correct match, competitive with reasoning models
4. **Pedagogical value**: Detailed step-by-step solutions with knowledge point extraction support genuine learning

## Limitations

- Evaluation focuses exclusively on the ProcessBench dataset, which may not fully represent the diversity of real student reasoning patterns
- ProcessBench labels only the first error in each solution; subsequent errors are not annotated
- Critic evaluation only (not end-to-end solver-critic pipeline) due to project scope
- Cost calculations based on current OpenAI API pricing (subject to change)

## Authors

**Hugo Lin** (lifanlin@stanford.edu) - Department of Statistics, Stanford University
**David Lyu** (dlyu@stanford.edu) - Department of Statistics, Stanford University

This project was developed for CS329T (Trustworthy Machine Learning: Building and Evaluating Agentic Systems), Autumn 2025, Stanford University.

## Citation

If you use ShuTong in your research, please cite:

```bibtex
@misc{lin2025shutong,
  title={ShuTong: Trustworthy Step-by-Step Math Problem Solving},
  author={Lin, Hugo and Lyu, David},
  year={2025},
  note={CS329T Final Project, Stanford University}
}
```

## License

This project is licensed under the AGPL-3.0 License - see the LICENSE file for details.

## Acknowledgments

We thank the CS329T teaching team, especially Professor John Mitchell and Professor Anupam Datta, for their valuable feedback and guidance throughout this project.

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent workflow orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM integration
- [SymPy](https://www.sympy.org/) - Symbolic mathematics
- [ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench) - Evaluation dataset
- [MATH Dataset](https://github.com/hendrycks/math) - Mathematical reasoning problems
