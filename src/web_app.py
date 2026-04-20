from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import numpy as np
import os

# 1. Initialize the app with a clean Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Add this line for Render/Production
server = app.server

# 2. Load the trained model and preprocessing artifacts
base_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(base_dir, '../artifacts')

# Load model and artifacts
try:
    model = joblib.load(os.path.join(artifacts_dir, 'model_lr.pkl'))
    scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.pkl'))
    feature_columns = joblib.load(os.path.join(artifacts_dir, 'feature_columns.pkl'))
    print("Model and artifacts loaded successfully!")
    print(f"Features expected: {len(feature_columns)}")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    model = None
    scaler = None
    feature_columns = None

# 3. Define the Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Stroke Risk Prediction Dashboard", className="text-center my-4"), width=12)
    ]),

    dbc.Row([
        # Left Column: Inputs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Patient Metrics"),
                dbc.CardBody([
                    # Gender & Age
                    dbc.Label("Gender"),
                    dcc.Dropdown(id='gender', options=['Male', 'Female'], value='Female', className="mb-3"),
                    
                    dbc.Label("Age", className="mt-2"),
                    html.Div([
                        dcc.Slider(id='age', min=0, max=100, step=1, value=30, 
                                   marks={0: '0', 20: '20', 40: '40', 60: '60', 80: '80', 100: '100'},
                                   className="mb-3"),
                        html.Div(id='age-value', className="text-center text-muted small")
                    ]),

                    # Health Conditions (Checkboxes)
                    dbc.Label("Health Conditions", className="mt-2"),
                    dbc.Checklist(
                        options=[
                            {"label": "Hypertension", "value": "hypertension"},
                            {"label": "Heart Disease", "value": "heart_disease"},
                        ],
                        value=[], id="health-conditions", inline=True, className="mb-3"
                    ),

                    # Work Type
                    dbc.Label("Work Type"),
                    dcc.Dropdown(
                        id='work_type', 
                        options=[
                            {'label': 'Private', 'value': 'Private'},
                            {'label': 'Self-employed', 'value': 'Self-employed'},
                            {'label': 'Children', 'value': 'children'},
                            {'label': 'Never worked', 'value': 'Never_worked'},
                            {'label': 'Government Job', 'value': 'Govt_job'}
                        ], 
                        value='Private', 
                        className="mb-3"
                    ),

                    # Residence Type
                    dbc.Label("Residence Type"),
                    dcc.Dropdown(
                        id='residence', 
                        options=['Urban', 'Rural'], 
                        value='Urban', 
                        className="mb-3"
                    ),

                    # Marital Status
                    dbc.Label("Ever Married"),
                    dcc.Dropdown(
                        id='ever_married', 
                        options=['Yes', 'No'], 
                        value='Yes', 
                        className="mb-3"
                    ),

                    # Numerical Inputs
                    dbc.Label("Average Glucose Level (mg/dL)"),
                    dbc.Input(id='glucose', type='number', min=50, max=300, step=1, value=100, className="mb-3"),
                    html.Div("Normal range: 70-140 mg/dL", className="text-muted small mb-2"),

                    dbc.Label("BMI (kg/m²)"),
                    dbc.Input(id='bmi', type='number', min=10, max=60, step=0.1, value=25, className="mb-3"),
                    html.Div("Normal range: 18.5-24.9", className="text-muted small mb-2"),

                    # Smoking Status
                    dbc.Label("Smoking Status"),
                    dcc.Dropdown(
                        id='smoking', 
                        options=[
                            {'label': 'Never smoked', 'value': 'never smoked'},
                            {'label': 'Currently smokes', 'value': 'smokes'},
                            {'label': 'Formerly smoked', 'value': 'formerly smoked'}
                        ], 
                        value='never smoked', 
                        className="mb-3"
                    ),

                    dbc.Button("Predict Risk", id="predict-btn", color="primary", className="w-100 mt-3")
                ])
            ], className="shadow")
        ], md=5),

        # Right Column: Results & Advice
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prediction Output"),
                dbc.CardBody([
                    html.Div(id="prediction-output", className="text-center h3 py-4"),
                    html.Hr(),
                    html.H5("Clinical Recommendations"),
                    html.Div(id="medical-advice", className="text-muted"),
                    html.Hr(),
                    html.H5("Risk Factors Identified"),
                    html.Div(id="risk-factors", className="small text-muted")
                ])
            ], className="shadow")
        ], md=7)
    ])
], fluid=True)

# Display current age value
@app.callback(
    Output("age-value", "children"),
    [Input("age", "value")]
)
def update_age_display(age):
    return f"Selected age: {age} years"

# 4. Helper function to prepare input data for the model
def prepare_input_data(gender, age, conditions, work_type, residence, ever_married, glucose, bmi, smoking):
    """Convert UI inputs to model-ready format"""
    
    if feature_columns is None:
        raise Exception("Model artifacts not loaded")
    
    # Create a dictionary with all features initialized to 0
    input_dict = {col: 0 for col in feature_columns}
    
    # Map categorical variables
    gender_map = {'Female': 0, 'Male': 1}
    input_dict['gender'] = gender_map.get(gender, 0)
    
    # Numerical values
    input_dict['age'] = float(age)
    input_dict['hypertension'] = 1 if 'hypertension' in conditions else 0
    input_dict['heart_disease'] = 1 if 'heart_disease' in conditions else 0
    input_dict['avg_glucose_level'] = float(glucose)
    input_dict['bmi'] = float(bmi)
    
    # Ever married: Yes=1, No=0
    input_dict['ever_married'] = 1 if ever_married == 'Yes' else 0
    
    # Residence type: Urban=1, Rural=0
    input_dict['Residence_type'] = 1 if residence == 'Urban' else 0
    
    # Work type (one-hot encoded)
    work_type_map = {
        'Private': 'work_type_Private',
        'Self-employed': 'work_type_Self-employed',
        'children': 'work_type_children',
        'Never_worked': 'work_type_Never_worked',
        'Govt_job': 'work_type_Govt_job'
    }
    work_col = work_type_map.get(work_type)
    if work_col and work_col in input_dict:
        input_dict[work_col] = 1
    
    # Smoking status (one-hot encoded)
    smoking_map = {
        'never smoked': 'smoking_status_never smoked',
        'smokes': 'smoking_status_smokes',
        'formerly smoked': 'smoking_status_formerly smoked'
    }
    smoking_col = smoking_map.get(smoking)
    if smoking_col and smoking_col in input_dict:
        input_dict[smoking_col] = 1
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Scale numerical features
    numeric_cols = ['age', 'avg_glucose_level', 'bmi']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    return input_df

# 5. Helper function to generate clinical advice (FIXED VERSION)
def get_clinical_advice(risk_score, risk_probability, age, glucose, bmi, has_hypertension, has_heart_disease, smoking):
    """Generate personalized medical advice based on risk factors"""
    
    advice_lines = []
    risk_factors = []
    
    # ALWAYS collect all risk factors regardless of prediction
    if age > 60:
        risk_factors.append(f"Age {age} (elevated risk for stroke)")
    elif age > 50:
        risk_factors.append(f"Age {age} (moderate risk factor)")
    
    if has_hypertension:
        risk_factors.append("Hypertension (key stroke risk factor)")
    
    if has_heart_disease:
        risk_factors.append("Heart disease (increases stroke risk)")
    
    if glucose > 140:
        risk_factors.append(f"High glucose level ({glucose:.1f} mg/dL)")
    elif glucose > 125:
        risk_factors.append(f"Pre-diabetic glucose level ({glucose:.1f} mg/dL)")
    
    if bmi > 30:
        risk_factors.append(f"BMI {bmi:.1f} (obesity increases stroke risk)")
    elif bmi > 25:
        risk_factors.append(f"BMI {bmi:.1f} (overweight)")
    elif bmi < 18.5:
        risk_factors.append(f"BMI {bmi:.1f} (underweight)")
    
    if smoking == 'smokes':
        risk_factors.append("Current smoker (significantly increases stroke risk)")
    elif smoking == 'formerly smoked':
        risk_factors.append("Former smoker (previous smoking increases risk)")
    
    # Now generate advice based on prediction
    if risk_score == 1:
        advice_lines.append("⚠️ **HIGH RISK DETECTED** - Immediate follow-up recommended.")
        
        advice_lines.append("\n**Recommendations:**")
        advice_lines.append("• Schedule a cardiovascular consultation within 2 weeks")
        advice_lines.append("• Monitor blood pressure daily")
        advice_lines.append("• Consider lifestyle modifications (diet, exercise)")
        advice_lines.append("• Discuss antiplatelet therapy with your doctor")
        
        # Add specific advice based on risk factors
        if has_hypertension:
            advice_lines.append("• Work with your doctor to control blood pressure")
        if smoking == 'smokes':
            advice_lines.append("• Consider smoking cessation programs")
        if glucose > 140:
            advice_lines.append("• Monitor blood glucose and consider diabetes screening")
        
    else:
        advice_lines.append("✅ **LOW RISK** - Continue preventive care.")
        
        advice_lines.append("\n**Recommendations:**")
        advice_lines.append("• Maintain healthy lifestyle (exercise, balanced diet)")
        advice_lines.append("• Regular blood pressure and glucose checks")
        advice_lines.append("• Follow up annually or as recommended by your physician")
        
        # Even for low risk, provide specific advice for existing conditions
        if has_hypertension:
            advice_lines.append("• Continue managing your blood pressure with your doctor")
        if has_heart_disease:
            advice_lines.append("• Continue heart disease management as prescribed")
        if smoking == 'smokes':
            advice_lines.append("• Smoking cessation would further reduce your risk")
        if bmi > 30:
            advice_lines.append("• Weight management could help reduce future risk")
    
    return "\n".join(advice_lines), risk_factors

# 6. Define the Callback (Logic)
@app.callback(
    [Output("prediction-output", "children"),
     Output("medical-advice", "children"),
     Output("risk-factors", "children")],
    [Input("predict-btn", "n_clicks")],
    [State("gender", "value"),
     State("age", "value"),
     State("health-conditions", "value"),
     State("work_type", "value"),
     State("residence", "value"),
     State("ever_married", "value"),
     State("glucose", "value"),
     State("bmi", "value"),
     State("smoking", "value")]
)
def run_prediction(n, gender, age, conditions, work_type, residence, ever_married, glucose, bmi, smoking):
    if n is None:
        return "Waiting for input...", "Enter patient details to see clinical advice.", ""
    
    if model is None:
        return "Model Error", "Model not loaded. Please check server logs.", ""
    
    # Prepare input data for the model
    try:
        # Validate inputs
        if age is None or glucose is None or bmi is None:
            return "Invalid Input", "Please fill in all fields.", ""
        
        # Prepare and predict
        input_df = prepare_input_data(gender, age, conditions, work_type, residence, ever_married, glucose, bmi, smoking)
        
        # Get prediction (0 = No Stroke, 1 = Stroke)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of stroke
        
        # Get clinical advice
        has_hypertension = 'hypertension' in conditions
        has_heart_disease = 'heart_disease' in conditions
        
        advice, risk_factors = get_clinical_advice(
            prediction, probability, age, glucose, bmi, 
            has_hypertension, has_heart_disease, smoking
        )
        
        # Format output based on prediction
        if prediction == 1:
            risk_percent = probability * 100
            output = html.Div([
                html.Span("HIGH RISK", style={"color": "#dc3545", "font-weight": "bold", "font-size": "2rem"}),
                html.Br(),
                html.Small(f"Stroke Probability: {risk_percent:.1f}%", style={"font-size": "14px", "color": "#dc3545"})
            ])
        else:
            risk_percent = probability * 100
            output = html.Div([
                html.Span("LOW RISK", style={"color": "#28a745", "font-weight": "bold", "font-size": "2rem"}),
                html.Br(),
                html.Small(f"Stroke Probability: {risk_percent:.1f}%", style={"font-size": "14px", "color": "#28a745"})
            ])
        
        # Format risk factors for display
        if risk_factors:
            risk_factors_display = html.Ul([html.Li(factor) for factor in risk_factors])
        else:
            risk_factors_display = "No major risk factors identified."
        
        return output, advice, risk_factors_display
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return "Prediction Error", f"Unable to generate prediction. Error: {str(e)}", ""

# 7. Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)