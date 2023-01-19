"""
A dashboard that predicts the disease of a user based off their symptoms and reports to the user some useful
information regarding various disease statistics

To do:
Mia - look for other classification models; look into HTML components to make it look more aesthetic; report to user the
accuracy of their chosen model: 'with ___ accuracy, _______ model predicts your disease as ____'

figure out how to put sankey on dashboard page
figure out why round() function breaks the code
"""

import pandas as pd
from Disease_Predictor import *
from sankey_copy import *
from map_viz import *
from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
pd.options.mode.chained_assignment = None

# Read in the data for the disease prediction
initial_df = pd.read_csv('disease_data.csv')

# Obtain the disease and symptoms DataFrame
dis_sym_df, sym_lst = get_prediction_df(initial_df)

# Create a DataFrame to use to produce our sankey diagram
sankey_df = make_df(initial_df)

# Obtain a list of all the diseases in our prediction data
disease = list(sankey_df['Disease'].unique())

# Create a list of options for the user to make a sankey diagram
sk_options = ['All'] + disease

# Read in the precaution data
prec_df = pd.read_csv('symptom_precaution.csv')

# Get the data needed to create the heat map
heat_df = pd.read_csv('grouped_2020.csv')

# Get the data needed to create the disease prevalence map
dis_prev_df = pd.read_csv('measures_2020.csv')

# Obtain a list of the diseases that can be used to create our disease prevelance map
disease_unique = list(dis_prev_df.Short_Question_Text.unique())

# Build an app to display sunspot data
app = Dash(__name__)

# Due to the implementation of tabs, suppress callback exceptions
app.config.suppress_callback_exceptions = True

# Define the components needed to style the dashboard
style = {'background': '#270980', 'text': '#270980', 'font-family': 'candara'}
# OTHER POTENTIAL COLORS: 7AD6EB or white

# Format the layout of the dashboard
app.layout = html.Div(style={'textAlign': 'center', 'fontWeight': 'bold'},
                      children=[
    html.H1('Interactive Dashboard for Disease Prediction and Disease Information Reporting', style={'backgroundColor':
                                                                                                         '#270980',
                                                                                                     'margin': '0',
                                                                                                     'color': 'white',
                                                                                                     'font-size': '200%'
                                                                                                     'padding-top:10px'}),
    dcc.Tabs(id='tabs', value='tab_1', children=[
        dcc.Tab(label='Introduction', value='tab_1', style={'color': style['text'], 'font-family': style['font-family'],
                                                            'background': 'white'}),
        dcc.Tab(label='Disease Prediction', value='tab_2', style={'color': style['background'], 'font-family': style['font-family'],
                                                                  'background':'white'}),
        dcc.Tab(label='Sankey Diagrams', value='tab_3', style={'color': style['background'], 'font-family': style['font-family'],
                                                               'background':'white'}),
        dcc.Tab(label='Disease Heat Map', value='tab_4', style={'color': style['background'], 'font-family': style['font-family'],
                                                          'background':'white'}),
        dcc.Tab(label='Disease Prevalence Map', value='tab_5', style={'color': style['background'], 'font-family': style['font-family'],
                                                                      'background':'white'})
    ]),
    html.Div(id='tabs_content')
])


# Make a decorator for the tabs and render the content of the tabs
@app.callback(
    Output('tabs_content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab_1':
        return html.Div([
            html.P('Welcome to the interactive dashboard for disease prediction and disease information reporting. \
                       Our goal is to inform you in regards to your personal medical concerns. This dashboard contains \
                       several interactive tabs which will allow you to gain a better understand of your diagnosis, as \
                       well as understand disease statistics within the US.', style={'margin': '100'}),
            html.P('This Dashboard performs several tasks. A Disease Prediction model allows you to insert your symptoms \
                       and obtain a diagnosis along with a prescribed treatment. Both a heat map and bubble map display \
                    disease prevalence based on the disease chosen, and a Sankey diagram links symptom to disease and vice-versa \
                       to show the frequency at which symptoms and diseases are correlated, and which disease share the same \
                       symptoms.', style={'margin': '100'}),
            html.H2('Data Sources', style={'text-align': 'left', 'font-family': style['font-family'],
                                           'padding-left': '75px'}),
            html.Li(html.A('Disease Prediction Data',
                           href="https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset",
                           target='_blank'),
                    style={'text-align': 'left', 'font-family': style['font-family'], 'font-size':'110%',
                           'padding-left': '75px'}),
            html.Li(html.A('Map Data',
                           href="https://chronicdata.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-\
                           Place-Data-202/eav7-hnsx",
                           target='_blank'),
                    style={'text-align': 'left', 'font-family': style['font-family'], 'font-size':'110%',
                           'padding-left': '75px'})
        ])
    elif tab == 'tab_2':
        return html.Div(style={'padding-top':'100px'}, children=[
            html.P('Please select the symptoms you are experiencing:'),
            dcc.Dropdown(sym_lst, multi=True, id='options_chosen'),
            html.P('Please select the algorithm you would like to use for this prediction:'),
            dcc.RadioItems(options=['Random Forest Classifier', 'Naive Bayes Model', 'K-Nearest Neighbors Classifier',
                                    'Logistic Regression Model'],
                           value='Random Forest Classifier',
                           id='radio_button',
                           style={'font-family':style['font-family']}),
            html.P('Please press the done button when you have finished entering your symptoms'),
            dcc.Checklist(['Done'], id='checklist', style={'font-family':style['font-family']}),
            html.Div(id='text_return'),
            html.P('Note: to create a new prediction, refresh the page')
        ])
    elif tab == 'tab_3':
        return html.Div([
            html.P('Please select the disease(s) you would like to see a Sankey Diagram of:'),
            dcc.Dropdown(sk_options, multi=True, id='disease_chosen'),
            dcc.Graph(id='sankey', style={'width': '100vw', 'height': '65vh'})

        ])
    elif tab == 'tab_4':
        return html.Div([
          html.P('Please select the disease you would like to see a heat map of:'),
          dcc.Dropdown(disease_unique, id='disease_list'),
          dcc.Graph(id='dis_heat_map', style={'width': '100vw', 'height': '77vh'})
        ])
    elif tab == 'tab_5':
        return html.Div([
            html.P('Please select the disease you would like to see a prevalence map of:'),
            dcc.Dropdown(disease_unique, id='disease_unique'),
            dcc.Graph(id='dis_prev_map', style={'width': '100vw', 'height': '65vh'})
        ])


# Create a decorator for tab 1: disease predictor
@app.callback(
    Output('text_return', 'children'),
    Input('options_chosen', 'value'),
    Input('checklist', 'value'),
    Input('radio_button', 'value')
)
def update_predictor(options_chosen, checklist, radio_button):
    if not options_chosen:
        raise PreventUpdate
    if not checklist:
        raise PreventUpdate

    user_syms = []
    appender = [user_syms.append(0.0) if sym not in options_chosen else user_syms.append(1.0) for sym in
                list(dis_sym_df.columns)[1:]]
    user_syms_df = pd.DataFrame(data=[user_syms], columns=list(dis_sym_df.columns)[1:])
    rfc_modl, nb_modl, knn_modl, lr_modl = predict_disease(initial_df)
    if radio_button == 'Random Forest Classifier':
        pred_disease = str(rfc_modl.predict(user_syms_df)[0])
    elif radio_button == 'K-Nearest Neighbors Classifier':
        pred_disease = str(knn_modl.predict(user_syms_df)[0])
    elif radio_button == 'Logistic Regression Model':
        pred_disease = str(lr_modl.predict(user_syms_df)[0])
    else:
        pred_disease = str(nb_modl.predict(user_syms_df)[0])

    prec_str = report_precautions(pred_disease, prec_df)

    markdown = dcc.Markdown(
        f"""
    ## **Your predicted disease:** {str(pred_disease)}
    
    ## **Your recommended precautions:** {prec_str}
        """, id='text_return')

    return markdown


# Create a decorator for tab 2: sankey diagram
@app.callback(
    Output('sankey', 'figure'),
    Input('disease_chosen', 'value'),
)
def update_predictor(disease_chosen):
    if not disease_chosen:
        raise PreventUpdate

    if disease_chosen == ['All']:
        disease_chosen = disease

    df_input = filter_input(sankey_df, 'Disease', disease_chosen)

    # make sankey diagrams and aggregate dataframe for selected columns
    fig = make_sankey_diagrams(df_input, 'Disease', 'Symptom', 'Val')

    return fig

# Create a decorator for tab 4: disease heat map
@app.callback(
    Output('dis_heat_map', 'figure'),
    Input('disease_list', 'value'),
)
def update_prev_fig(disease_unique):
    # Fig returned corresponds to filtered disease
    heat_fig = make_map2(heat_df, disease_unique)
    return heat_fig

# Create a decorator for tab 5: disease prevalence diagram
@app.callback(
    Output('dis_prev_map', 'figure'),
    Input('disease_unique', 'value'),
)
def update_prev_fig(disease_unique):
    # Fig returned corresponds to filtered disease
    prevalence_fig = make_map(dis_prev_df, disease_unique)
    return prevalence_fig


if __name__ == '__main__':
    # Run the app
    app.run_server(debug=True)