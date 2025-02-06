# Formula-1-Predictor
![alt text](Extras/Header_ReadMe.jpg)

---

# Formula 1 Grand Prix Predictor

## Overview
The **Formula 1 Grand Prix Predictor** project utilizes machine learning to predict the outcomes of Formula 1 (F1) races. By training a **Random Forest** model with data scraping techniques and leveraging **Pandas** for data manipulation and **Scikit Learn** for model building, this project aims to predict race results based on historical data.

## Features
- **Random Forest Model**: A supervised machine learning algorithm used to predict race outcomes.
- **Data Scraping**: Collected historical Formula 1 data using web scraping techniques for feature engineering.
- **Data Processing**: Used Pandas to clean, manipulate, and prepare data for analysis.
- **Model Evaluation**: Evaluates model performance using appropriate metrics such as accuracy and precision.
- **Visualizations**: Displays key insights and predictions via data visualizations.

## Technologies
- **Python**: The primary programming language used for the project.
- **Pandas**: For data manipulation and cleaning.
- **Scikit Learn**: For building and training the machine learning model.
- **BeautifulSoup / Requests**: For scraping race results and related historical data from the web.
- **Matplotlib/Seaborn**: For data visualization.

## Setup & Installation

### Prerequisites
- Python 3.7+

### Steps to run the project:
1. Clone the repository:

```bash
git clone https://github.com/Adri-ayala/Formula-1-Predictor.git
cd Formula-1-Predictor
```

1. To scrape historical race data, run the scraper script:

```bash
Formula1_2024_Teams_Scraped.py
```

2. Train the model using the following script and make predictions using the trained model:
```bash
F1_Predictor.py
```
## Model Evaluation
The model's performance can be evaluated using standard classification metrics:

- **Accuracy**: 96%
- **Precision and Recall**: 71%.

## Data Sources
- [Formula 1 Official Website](https://www.formula1.com/)
- [Ergast API](https://ergast.com/mrd/)

## Contributing
Contributions are welcome! If you find bugs, want to improve the model, or add new features, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
