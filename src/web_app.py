from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# 1. Initialize the app with a clean Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Add this line for Render/Production
server = app.server

# 2. Define the Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Health Risk Prediction Dashboard", className="text-center my-4"), width=12)
    ]),

    dbc.Row([
        # Left Column: Inputs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Patient Metrics"),
                dbc.CardBody([
                    # Gender & Age
                    dbc.Label("Gender"),
                    dcc.Dropdown(id='gender', options=['Male', 'Female', 'Other'], value='Female', className="mb-3"),
                    
                    dbc.Label("Age"),
                    dcc.Slider(id='age', min=0, max=100, step=1, value=30, marks={i: str(i) for i in range(0, 101, 20)}, className="mb-3"),

                    # Health Conditions (Checkboxes)
                    dbc.Checklist(
                        options=[
                            {"label": "Hypertension", "value": 1},
                            {"label": "Heart Disease", "value": 2},
                        ],
                        value=[], id="health-conditions", inline=True, className="mb-3"
                    ),

                    # Numerical Inputs
                    dbc.Label("Average Glucose Level"),
                    dbc.Input(id='glucose', type='number', min=50, max=240, value=100, className="mb-3"),

                    dbc.Label("BMI"),
                    dbc.Input(id='bmi', type='number', min=20, max=50, value=25, className="mb-3"),

                    # Smoking Status
                    dbc.Label("Smoking Status"),
                    dcc.Dropdown(id='smoking', options=['Never', 'Active', 'Stopped'], value='Never', className="mb-3"),

                    dbc.Button("Predict Result", id="predict-btn", color="primary", className="w-100 mt-3")
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
                    html.H5("Medical Comments & Advice"),
                    html.Div(id="medical-advice", className="text-muted italic")
                ])
            ], className="shadow")
        ], md=7)
    ])
], fluid=True)

# 3. Define the Callback (Logic)
@app.callback(
    [Output("prediction-output", "children"),
     Output("medical-advice", "children")],
    [Input("predict-btn", "n_clicks")],
    [State("gender", "value"),
     State("age", "value"),
     State("health-conditions", "value"),
     State("glucose", "value"),
     State("bmi", "value"),
     State("smoking", "value")]
)
def run_prediction(n, gender, age, conditions, glucose, bmi, smoking):
    if n is None:
        return "Waiting for input...", "Enter patient details to see clinical advice."
    
    # Logic to process checkboxes
    has_hypertension = 1 in conditions
    has_heart_disease = 2 in conditions
    
    # PLACEHOLDER: This is where we would call our model.predict()
    
    prediction_score = (age * 0.1) + (glucose * 0.05)
    
    # Mock result logic
    if prediction_score > 15:
        return "HIGH RISK", "Recommendation: Follow up with immediate cardiovascular screening."
    return "LOW RISK", "Patient appears within normal parameters for these metrics."

if __name__ == '__main__':
    app.run(debug=True)
