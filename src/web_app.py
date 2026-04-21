from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import numpy as np
import os

# 1. Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# 2. Load models and artifacts
base_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(base_dir, '../artifacts')

try:
    stroke_model = joblib.load(os.path.join(artifacts_dir, 'model_lr.pkl'))
    cluster_model = joblib.load(os.path.join(artifacts_dir, 'model_km.pkl'))
    scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.pkl'))
    feature_columns = joblib.load(os.path.join(artifacts_dir, 'feature_columns.pkl'))
except Exception as e:
    print(f"Error loading artifacts: {e}")
    stroke_model = cluster_model = scaler = feature_columns = None

# Risk Levels from Clustering
RISK_LEVELS = {
    0: {"level": "MODERATE-HIGH", "color": "#fd7e14", "priority": 2,
        
        "recommendations": ["Weight management program", "30-45 min daily physical activity"]},
    1: {"level": "MODERATE", "color": "#ffc107", "priority": 3,
        
        "recommendations": ["Regular health screenings", "Annual cardiovascular check-up"]},
    2: {"level": "LOW", "color": "#28a745", "priority": 4,
       
        "recommendations": ["Maintain healthy lifestyle", "Avoid smoking"]},
    3: {"level": "HIGH", "color": "#dc3545", "priority": 1,
        
        "recommendations": ["URGENT: Schedule comprehensive health evaluation", "Medication review with doctor"]}
}

# 3. Layout with 3-Column Inputs and Compact Results
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Stroke and Health Risk Assessment", className="text-center my-3"), width=12)
    ]),
    
    # Input Section
    dbc.Card([
        dbc.CardHeader("Patient Profile & Clinical Inputs"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Gender"),
                    dcc.Dropdown(id='gender', options=['Male', 'Female'], value='Female', className="mb-2"),
                    dbc.Label("Age"),
                    dcc.Slider(id='age', min=0, max=100, step=1, value=30, className="mb-2"),
                    html.Div(id='age-value', className="text-muted small mb-2"),
                ], md=4),
                dbc.Col([
                    dbc.Label("Avg Glucose (mg/dL)"),
                    dbc.Input(id='glucose', type='number', value=100, className="mb-2"),
                    dbc.Label("BMI (kg/m²)"),
                    dbc.Input(id='bmi', type='number', value=25, className="mb-2"),
                    dbc.Label("Health Conditions"),
                    dbc.Checklist(
                        options=[{"label": "Hypertension", "value": "hypertension"},
                                 {"label": "Heart Disease", "value": "heart_disease"}],
                        value=[], id="health-conditions", inline=True
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Label("Work Type"),
                    dcc.Dropdown(id='work_type', options=[
                        {'label': 'Private Sector', 'value': 'Private'},
                        {'label': 'Self-employed', 'value': 'Self-employed'},
                        {'label': 'Child Caretaking', 'value': 'children'},
                        {'label': 'Never worked', 'value': 'Never_worked'},
                        {'label': 'Government / Public Sector', 'value': 'Govt_job'}
                    ], value='Private', className="mb-2"),
                    dbc.Label("Smoking Status"),
                    dcc.Dropdown(id='smoking', options=[
                        {'label': 'Never smoked', 'value': 'never smoked'},
                        {'label': 'Currently smokes', 'value': 'smokes'},
                        {'label': 'Formerly smoked', 'value': 'formerly smoked'}
                    ], value='never smoked', className="mb-3"),
                    dbc.Button("Analyze Health Profile", id="predict-btn", color="primary", className="w-100 shadow-sm")
                ], md=4),
            ])
        ])
    ], className="shadow-sm mb-4"),

    # Results Row
    dbc.Row([
        # Stroke Probability & Risk Factors (Half/Half split)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stroke Risk Analysis"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="prediction-output", className="text-center h4 py-2"),
                           # html.Small("Risk Likelihood", className="text-muted d-block text-center")
                        ], width=6, style={"border-right": "1px solid #dee2e6"}),
                        dbc.Col([
                            html.H6("Identified Risk Factors:", className="mb-2"),
                            html.Div(id="risk-factors", className="small text-danger")
                        ], width=6),
                    ])
                ])
            ], className="shadow-sm h-100")
        ], md=6),
        
        # Clinical Assessment
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Genral Health Risk Assessment (Clustering)"),
                dbc.CardBody([
                    html.Div(id="risk-level-output", className="mb-3"),
                    html.H6("Recommendations:"),
                    html.Div(id="risk-recommendations", className="small text-muted")
                ])
            ], className="shadow-sm h-100")
        ], md=6),
    ])
], fluid=True)

# --- Logic & Callbacks ---

@app.callback(Output("age-value", "children"), [Input("age", "value")])
def update_age_display(age):
    return f"Selected age: {age} years"

def prepare_input_data(gender, age, conditions, work_type, residence, ever_married, glucose, bmi, smoking):
    # This remains consistent with your previous feature engineering logic
    input_dict = {col: 0 for col in feature_columns}
    input_dict['gender'] = 0 if gender == 'Female' else 1
    input_dict['age'] = float(age)
    input_dict['hypertension'] = 1 if 'hypertension' in conditions else 0
    input_dict['heart_disease'] = 1 if 'heart_disease' in conditions else 0
    input_dict['avg_glucose_level'] = float(glucose)
    input_dict['bmi'] = float(bmi)
    input_dict['ever_married'] = 1 # Defaulted for brevity
    input_dict['Residence_type'] = 1 
    
    work_map = {'Private': 'work_type_Private', 'Self-employed': 'work_type_Self-employed', 
                'children': 'work_type_children', 'Never_worked': 'work_type_Never_worked', 
                'Govt_job': 'work_type_Govt_job'}
    if work_map.get(work_type) in input_dict:
        input_dict[work_map[work_type]] = 1
    
    smoke_map = {'never smoked': 'smoking_status_never smoked', 'smokes': 'smoking_status_smokes', 
                 'formerly smoked': 'smoking_status_formerly smoked'}
    if smoke_map.get(smoking) in input_dict:
        input_dict[smoke_map[smoking]] = 1
    
    input_df = pd.DataFrame([input_dict])
    input_df[['age', 'avg_glucose_level', 'bmi']] = scaler.transform(input_df[['age', 'avg_glucose_level', 'bmi']])
    return input_df

@app.callback(
    [Output("prediction-output", "children"),
     Output("risk-factors", "children"),
     Output("risk-level-output", "children"),
     Output("risk-recommendations", "children")],
    [Input("predict-btn", "n_clicks")],
    [State("gender", "value"), State("age", "value"), State("health-conditions", "value"),
     State("work_type", "value"), State("glucose", "value"), State("bmi", "value"), State("smoking", "value")]
)
def run_prediction(n, gender, age, conditions, work_type, glucose, bmi, smoking):
    if n is None:
        return "Waiting...", "", "Assessment pending", "Inputs required"
    
    input_df = prepare_input_data(gender, age, conditions, work_type, "Urban", "Yes", glucose, bmi, smoking)
    
    # 1. Stroke Probability Categorization
    prob = stroke_model.predict_proba(input_df)[0][1] * 100
    if prob < 20:
        cat, color = "LOW", "#28a745"
    elif prob < 40:
        cat, color = "MODERATE", "#ffc107"
    else:
        cat, color = "HIGH", "#dc3545"
    
    stroke_display = html.Div([
        html.Span(cat, style={"color": color, "font-weight": "bold", "font-size": "2rem"}),
        html.Br(),
        html.Small(f"Probability: {prob:.1f}%")
    ])
    
    # 2. Risk Factor Logic
    factors = []
    if age > 55: factors.append(f"Age (>55)")
    if 'hypertension' in (conditions or []): factors.append("Hypertension")
    if 'heart_disease' in (conditions or []): factors.append("Heart Disease") # Explicit check
    if glucose > 140: factors.append("Hyperglycemia")
    if bmi > 25: factors.append("Overweight/Obese")
    if smoking == 'smokes': factors.append("Active Smoker")
    
    factors_display = html.Ul([html.Li(f) for f in factors]) if factors else html.Em("...")

    # 3. Clustering Logic
    cluster = cluster_model.predict(input_df)[0]
    risk_info = RISK_LEVELS.get(cluster)
    risk_display = html.Div([
        html.Span(risk_info["level"], style={"color": risk_info["color"], "font-weight": "bold", "font-size": "1.5rem"}),
        html.Br(),
        html.Small(f"Priority {risk_info['priority']}")
    ])
    recs = html.Ul([html.Li(r) for r in risk_info["recommendations"]])
    
    return stroke_display, factors_display, risk_display, recs

if __name__ == '__main__':
    app.run(debug=False, port=int(os.environ.get('PORT', 8050)))