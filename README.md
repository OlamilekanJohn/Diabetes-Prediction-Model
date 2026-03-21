# 🩺 Diabetes Risk Predictor

> **Can a machine help catch diabetes before it's too late?**
> This project says yes — and makes it available to everyone.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange?logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-Containerised-blue?logo=docker)
![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-yellow?logo=amazonaws)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 What Is This Project?

Type 2 diabetes affects hundreds of millions of people worldwide — and many don't know they're at risk until it's too late. Early detection changes outcomes dramatically.

This project is a **machine learning system** that predicts whether someone is at risk of developing Type 2 Diabetes, based on health metrics. What makes it different from a typical university project?

- It's **live and accessible** — no coding needed. Anyone can visit the web interface, enter their details, and get a result instantly.
- It's built with a **safety-first mindset** — the model is tuned to minimise missed diagnoses, not just chase accuracy scores.
- It's **production-grade** — deployed on cloud infrastructure with automated testing, containerisation, and monitoring.

---

## 🎯 The Problem It Solves

| Challenge | How This Project Addresses It |
|---|---|
| Diabetes often goes undetected | Predicts risk before symptoms appear |
| Medical tools require clinical access | Anyone can use this via a simple web page |
| High false negatives are dangerous | Model is tuned to catch as many at-risk cases as possible |
| ML projects rarely leave the notebook | This is fully deployed and live in the cloud |

---

## ✨ Key Features

**🔬 Rigorous Machine Learning Pipeline**
The model wasn't just trained — it was built carefully. The process included inspecting and cleaning real data, engineering meaningful features, selecting only the most useful inputs, and tuning the model's settings to achieve the best possible results. Results were validated using K-Fold cross-validation, a technique that tests the model across multiple data splits to ensure it generalises well.

**🤖 Two Models, One Goal**
Alongside the primary Logistic Regression model, a second unsupervised model (K-Means Clustering) was built independently — with no labels or guidance — to see if it could find the same patterns on its own. It did. This gives extra confidence that the patterns the model learns are real, not accidental.

**⚖️ Safety-First: Prioritising Recall**
In healthcare, missing a real diagnosis (a false negative) is far worse than an unnecessary follow-up (a false positive). This model is deliberately tuned to catch as many at-risk individuals as possible, accepting that some healthy individuals may be flagged — because that is the safer trade-off.

**🌐 Live Web App — No Coding Required**
The model is wrapped in a clean web interface built with Gradio. Any user can open a browser, enter their health metrics, and receive a prediction and personalised health recommendations in seconds.

**☁️ Fully Cloud-Deployed**
This isn't running on a laptop. The application is deployed on AWS and stays live continuously. Every code change automatically triggers a rebuild and redeployment.

---

## 📁 Project Structure

```
First deployed Project/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions — automated build & deploy pipeline
│
├── artifacts/
│   ├── feature_columns.json        # Saved feature schema used at inference time
│   └── preprocessing.pk1           # Serialised preprocessing pipeline
│
├── configs/                        # Configuration files (model settings, paths, etc.)
│
├── data/
│   ├── external/                   # Any third-party or supplementary data
│   ├── processed/
│   │   └── diabetes_processed.csv  # Cleaned, feature-engineered dataset
│   └── raw/
│       └── diabetes_unclean.csv    # Original, unmodified source data
│
├── great_expections/               # Legacy data validation config (replaced by Pandera)
│
├── mlruns/                         # MLflow experiment tracking logs
│
├── notebooks/
│   └── 250193855 Code File-Copy1.ipynb   # Full ML experimentation notebook
│
├── scripts/                        # Utility and helper scripts
│
├── src/
│   └── app/
│       ├── __init__.py
│       ├── app.py                  # Gradio web UI entry point
│       └── main.py                 # Application entry point
│   └── data/
│       ├── __init__.py
│       ├── load_data.py            # Data loading logic
│       └── preprocess.py          # Cleaning & preprocessing pipeline
│   └── features/
│       ├── __init__.py
│       └── features.py             # Feature engineering & encoding
│   └── models/
│       ├── __init__.py
│       ├── evaluate.py             # Model evaluation metrics
│       ├── train.py                # Model training script
│       └── tune.py                 # Hyperparameter tuning
│   └── serving/
│       ├── __init__.py
│       ├── model/                  # Saved model artefacts for serving
│       └── inference.py            # Prediction logic — called by the API
│   └── utils/
│       ├── __init__.py
│       ├── utils.py                # Shared utility functions
│       ├── validate_data_3.py      # Earlier iteration of data validation
│       └── validate_data.py        # Pandera-based data validation (current)
│
├── tests/                          # Unit and integration tests
│
├── Dockerfile                      # Container definition
└── requirement.txt                 # Python dependencies
```

---

## 🏗️ How It's Built

Here's a plain-English overview of the full system — from a user's click to a prediction:

```
User opens browser
       ↓
Application Load Balancer (AWS) receives the request on port 80
       ↓
Routes traffic to a running container (AWS ECS Fargate — serverless)
       ↓
Container runs inference.py via the Gradio UI or REST API
       ↓
Model processes the inputs and returns a prediction
       ↓
User sees their risk score + personalised recommendations
```

### Infrastructure at a Glance

| Layer | Technology | Purpose |
|---|---|---|
| **Model** | Scikit-learn (Logistic Regression) | Makes the prediction |
| **API** | REST API (`POST /predict`) | Lets other systems call the model programmatically |
| **Web UI** | Gradio (`/ui`) | Browser-based interface for end users |
| **Containerisation** | Docker | Packages the app so it runs identically anywhere |
| **Cloud Hosting** | AWS ECS Fargate | Runs containers without managing servers |
| **Traffic Management** | AWS Application Load Balancer (HTTP:80 → HTTP:8000) | Routes users to a healthy running instance |
| **Security** | AWS Security Groups | ALB accepts inbound port 80; containers only accept traffic from the ALB on port 8000 |
| **Monitoring** | AWS CloudWatch | Logs all container output and ECS service events |
| **CI/CD** | GitHub Actions (`ci.yml`) | Automates build, test, and deploy on every push to `main` |
| **Container Registry** | Docker Hub | Stores built images ready for deployment |

---

## 🚀 Deployment Flow (How Updates Go Live)

One of the goals of this project was **reliable, repeatable delivery** — meaning a code change should go from laptop to live production in a controlled, automated way.

Here's exactly what happens when a change is pushed:

```
1. Developer pushes code to the main branch on GitHub
         ↓
2. GitHub Actions (ci.yml) automatically builds a new Docker image
         ↓
3. The image is pushed to Docker Hub
         ↓
4. AWS ECS is triggered to force a new deployment
         ↓
5. The Load Balancer runs health checks — hitting / on port 8000
         ↓
6. Once the new container passes health checks, traffic is switched to it
         ↓
7. Users call POST /predict or open the Gradio UI at /ui via the ALB DNS
```

No manual uploading. No "works on my machine." Every deployment is tested before traffic reaches it.

---

## 🧪 The Experimentation Notebook

Before building the production system, extensive experimentation was carried out in a Jupyter Notebook (`notebooks/250193855 Code File-Copy1.ipynb`). This notebook is included in the repository and covers every step of the ML journey:

- **Dataset inspection** — understanding what the data looks like and spotting issues early
- **Data cleaning** — handling missing values, outliers, and inconsistencies
- **Feature engineering** — creating more informative inputs from raw data
- **Feature encoding** — converting categorical data into a format the model can use
- **Model training** — building and fitting the Logistic Regression model
- **K-Fold cross-validation** — rigorously testing model performance across multiple data splits
- **Feature selection** — identifying which inputs actually matter for prediction
- **Hyperparameter tuning** — finding the model settings that perform best
- **K-Means Clustering** — unsupervised pattern discovery used as independent validation

All key findings, charts, and conclusions are documented inside the notebook.

---

## 🔧 Challenges & How They Were Solved

Real projects hit real problems. Here's what went wrong and how it was fixed:

**1. Python couldn't find internal modules**

When the project grew beyond a single file, Python couldn't locate utility functions and validation scripts stored in subfolders like `src/utils`, `src/data`, and `src/serving`.

*Fix:* Added `__init__.py` files to each subfolder, which tells Python to treat them as proper packages. A small change with a big impact — and now visible throughout the project structure.

---

**2. The data validation library crashed in production**

The original plan was to use **Great Expectations** (still visible in the `great_expections/` folder) to validate incoming data before it reached the model. During testing, it crashed due to incompatibilities between its internal API and the version of Pandas being used — specifically around the `ge.dataset.PandasDataset` action.

*Fix:* Switched to **Pandera** (`src/utils/validate_data.py`), a simpler and more stable library that handled the same validation tasks cleanly. The data processing pipeline was also reordered so that data is cleaned *before* validation — reducing the chance of cascading failures downstream. MLflow experiment tracking was adjusted accordingly.

---

## 🛠️ Running It Locally

### Requirements
- Python 3.8+
- Docker (optional, for containerised setup)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/diabetes-risk-predictor.git
cd diabetes-risk-predictor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirement.txt

# 4. Start the application
python src/app/main.py
```

Then open your browser and go to `http://localhost:8000/ui`

### Using Docker

```bash
docker build -t diabetes-risk-predictor .
docker run -p 8000:8000 diabetes-risk-predictor
```

---

## 📡 API Reference

The model is also available programmatically via a REST API.

### `POST /predict`

Send health metrics, get a risk prediction back.

**Example request:**
```json
{
  "glucose": 148,
  "bmi": 33.6,
  "age": 50,
  "blood_pressure": 72,
  "insulin": 0,
  "skin_thickness": 35,
  "diabetes_pedigree": 0.627,
  "pregnancies": 6
}
```

**Example response:**
```json
{
  "risk": "High",
  "probability": 0.82,
  "recommendations": [
    "Consult a healthcare professional promptly.",
    "Increase physical activity to at least 150 minutes per week.",
    "Reduce intake of refined carbohydrates and sugars."
  ]
}
```

---

## 🤝 Contributing

Contributions are welcome. To get involved:

1. Fork this repository
2. Create a branch for your change (`git checkout -b feature/your-idea`)
3. Make your changes and commit them (`git commit -m 'Describe your change'`)
4. Push the branch (`git push origin feature/your-idea`)
5. Open a Pull Request and describe what you've done

For larger changes, please open an issue first so we can discuss the approach before you invest time building it.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute.

---

> ⚕️ **Medical Disclaimer:** This tool is built for educational and informational purposes only. It is not a medical device and should not be used as a substitute for professional medical advice, diagnosis, or treatment. If you have concerns about your health, please speak to a qualified healthcare professional.
