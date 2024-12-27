"""

-----------------------------------------------------------------
Dashboard : Assignment 2 - Magazine Recommendation system - Dashboard

Module Leader: Ming Jiang
Student Name: Kwok Chi Kiu
Student ID: 239669700
-----------------------------------------------------------------

"""

#------ Import library ------
import warnings
warnings.filterwarnings('ignore')

import datetime
from datetime import date
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle #=> for save dictionary 

#--- Regular expression library ---
import re

#--- Dash library ---
import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html, dash_table, State

#--- Diagram library ---
import plotly.graph_objects as go
import plotly.express as px

#--- Model library ---
from surprise import dump
#------ end Import library ------

#------ Common config ------
start_service_model = "production" # debug / production

show_number_of_sample_user = 20 #--- show number of sample user for "Manage User Recommendation" page ---
show_top_number_of_review = 40 #--- show number of user review ---
top_n_high_rate_and_review_item = 20
show_number_of_user_reviewed_items = 8
show_number_recommendation_items = 12

dataset_folder = "./datasets/" #--- datasets folder ---
item_json_file = dataset_folder + "/meta_Magazine_Subscriptions.jsonl" #--- magazine item data file ---
review_json_file = dataset_folder + "/Magazine_Subscriptions.jsonl" #--- user review data file ---

model_fullpath = './export_model/surprise_model' #--- model full path ---
log_userfile = './log/saved_log.pkl' #--- log user behavior file ---

today = date.today()
start_date = date(2022, 2, 1)
end_date = today
member_default_password = "1234" #==> default member password

admin_name = "admin" #==> default admin user name
admin_password = "1234" #==> default admin password
#------ end Common config ------

#--- Global variable ---
total_click_item = 0
total_recommendation_item = 0
dict_recommendation_item = defaultdict(dict)
dict_recommendation_item_click_rate = defaultdict(dict)
dict_recommendation_item_buy = defaultdict(dict)
log_dict = defaultdict(dict)
admin_user_id = ""
login_user_id = ""
global_href = ""
global_message = ""
global_pathname = ""
#--- end Global variable ---


print("--- Start Dashboard Service ---")

#------ Customize CSS style ------
white_font = {"color" : "white"}
red_font = {"color" : "red"}
green_font = {"color" : "green"}
black_font = {"color" : "black"}
font_10 = {"font-size" : "10px"}
font_13 = {"font-size" : "13px"}
font_15 = {"font-size" : "15px"}
font_16 = {"font-size" : "16px"}
font_20 = {"font-size" : "20px"}
font_30 = {"font-size" : "30px"}
font_40 = {"font-size" : "40px"}
font_60 = {"font-size" : "60px"}
bold = {"font-weight" : "bold"}
bg_black = {"background-color" : "#000"}
bg_grey = {"background-color" : "#cfcfcf"}
bg_brown = {"background-color" : "#faebd7"}
bg_orange = {"background-color" : "#ffc107"}
text_align_center = {"text-align" : "center"}
text_align_left = {"text-align" : "left"}
product_title_text = {
  "width" : "242px",
  "overflow" : "hidden",
  "display" : "inline-block",
  "text-overflow" : "ellipsis",
  "white-space" : "nowrap",
}
pad_0 = {"padding" : "0px"}
pad_5 = {"padding" : "5px"}
pad_10 = {"padding" : "10px"}
pad_top_20 = {"padding-top" : "20px"}
pad_top_5 = {"padding-top" : "5px"}
margin_10 = {"margin" : "10px"}
margin_top_10 = {"margin-top" : "10px"}
margin_top_20 = {"margin-top" : "20px"}
margin_top_5 = {"margin-top" : "5px"}
white_border = {"border" : "1px #FFF solid"}
grey_border = {"border" : "1px solid #c1c1c1"}
recommend_item_frame = {"overflow-y": "scroll", "height": "251px"}
high_rate_and_review_item_frame = {"overflow-y": "scroll", "height": "500px"}
hidden_html = {"display" : "none"}
show_html =  {"display" : "block"}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "display": "block",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sq_dash_frame = {}
sq_dash_frame.update(white_font)
sq_dash_frame.update(font_16)
sq_dash_frame.update(text_align_center)

sq_dash = {}
sq_dash.update(bg_black)
sq_dash.update(pad_10)

re_dash_frame = {}
re_dash_frame.update(pad_10)

re_dash_left_frame = {}
re_dash_left_frame.update(bg_orange)
re_dash_left_frame.update(font_60)
re_dash_left_frame.update(pad_10)
re_dash_left_frame.update(pad_top_20)
re_dash_left_frame.update(text_align_center)

re_dash_right_frame = {}
re_dash_right_frame.update(bg_orange)
re_dash_right_frame.update(pad_10)

re_dash_left_small_frame = {}
re_dash_left_small_frame.update(bg_orange)
re_dash_left_small_frame.update(font_30)
re_dash_left_small_frame.update(pad_10)
re_dash_left_small_frame.update(text_align_center)

re_dash_right_small_frame = {}
re_dash_right_small_frame.update(bg_orange)
re_dash_right_small_frame.update(pad_top_5)

msq_dash_frame = {}
msq_dash_frame.update(margin_top_20)
msq_dash_frame.update(show_html)

msq_dash_frame_hidden = {}
msq_dash_frame_hidden.update(hidden_html)

msq_dash = {}
msq_dash.update(pad_10)
msq_dash.update(grey_border)

list_item_frame = {}
list_item_frame.update(pad_10)
list_item_frame.update(text_align_center)

dash_small_title = {}
dash_small_title.update(bold)
dash_small_title.update(margin_top_5)
dash_small_title.update(pad_0)

user_recommendation_title = {}
user_recommendation_title.update(bg_brown)
user_recommendation_title.update(pad_10)
user_recommendation_title.update(font_20)
#------ end Customize CSS style ------

#------ All Common function ------

#--- Load Machine learning Model ---
def load_model(model_fullpath):
    user_predict_result, trained_model = dump.load(model_fullpath)    
    return user_predict_result, trained_model

#--- Save before record function ---
def save_log(dictionary, log_filepath):
    with open(log_filepath, 'wb') as f:
        pickle.dump(dictionary, f)    

#--- Load before record function ---
def load_log(log_filepath):
    loaded_dict = defaultdict(dict)
    try:
        with open(log_filepath, 'rb') as f:
            loaded_dict = pickle.load(f)
    except:
        pass
    return loaded_dict
    
#--- set user review sentiment ---
def set_user_review_sentiment(rating):
    if(rating < 3):
        return "Negative"
    elif(rating == 3):
        return "Neutral"
    elif(rating > 3):
        return "Postive"

#--- get item large image ---
def get_item_large_image(image):
    return image[0]["large"]

#--- Convert list to text ---
def convert_list_to_text(text):
    return "".join(text)

#--- Convert category to text ---
def convert_category_to_text(category):
    return ",".join(category)

#--- Get item from json ---
def get_item(item_json_file):
    item_df = pd.read_json(item_json_file, lines=True)
    item_df["large_image"] = item_df["images"].apply(lambda x : get_item_large_image(x))
    item_df["description"] = item_df["description"].apply(lambda x : convert_list_to_text(x))
    item_df["categories"] = item_df["categories"].apply(lambda x : convert_category_to_text(x))
    
    return item_df

#--- Get review from json ---
def get_review(review_json_file):
    review_df = pd.read_json(review_json_file, lines=True)
    review_df["date_year"] = review_df["timestamp"].dt.year
    review_df["date_month"] = review_df["timestamp"].dt.month
    review_df["date_day"] = review_df["timestamp"].dt.day
    review_df["date_week"] = review_df["timestamp"].dt.dayofweek
    review_df["date"] = review_df["timestamp"].dt.date
    review_df["year_month"] = review_df["date_year"].astype(str) + "-" + review_df["date_month"].astype(str)
    review_df["user_sentiment"] = review_df["rating"].apply(lambda x : set_user_review_sentiment(x))
    
    review_min_and_max_date = review_df['timestamp'].agg(['min', 'max'])
    min_date = review_min_and_max_date["min"]
    max_date = review_min_and_max_date["max"]
    max_date = date(max_date.year, max_date.month, max_date.day)
    
    return review_df, min_date, max_date

#--- Get total number of user ---
def get_total_number_of_user(review_df):
    df_user_id = review_df["user_id"].value_counts().reset_index()
    total_user = len(df_user_id)
    
    return df_user_id, total_user

#--- Get total number of item ---
def get_total_number_of_item(item_df):
    return len(item_df["parent_asin"].value_counts())

#--- Get total number of review message ---
def get_total_number_of_review(review_df):
    return len(review_df)

#--- Get some user for dropbox ---
def get_user_for_select(review_df):
    all_user_df = review_df["user_id"].value_counts().reset_index()
    all_user_df = all_user_df["user_id"].iloc[0:show_number_of_sample_user]
    return list(all_user_df)

#--- Get high top rate and review item ---
def get_top_high_rate_and_review_item(review_df, item_df, n=10):
    top_high_rate_item_df = review_df.groupby(by=["asin"]).agg({"rating" : "mean", "user_id" : "count"}).reset_index()
    top_high_rate_item_df.columns = ["item_id", "rating", "total_review"]
    top_high_rate_item_df = top_high_rate_item_df.sort_values(by=["rating", "total_review"], ascending=False)
    top_high_rate_item_df = top_high_rate_item_df.head(n)

    top_high_rate_item_df = pd.merge(left=top_high_rate_item_df, right=item_df, left_on='item_id', right_on='parent_asin', how="left")

    return top_high_rate_item_df[["item_id", "title", "rating",  "total_review"]]

#--- Get top n recommendation items ---
def get_top_n_recommendation_items(predict_result, item_df, user_id, n=5):    
    predict_rating_item_id = {}
    #-- loop for predictions result ---
    for uid, iid, true_r, est, _ in predict_result:
        if (uid==user_id):
            predict_rating_item_id[iid] = est # est : predict user rating
            
    #--- sorting high rating item id --- 
    predict_rating_item_id = sorted(predict_rating_item_id.items(), key=lambda kv: kv[1], reverse=True)[:n]
    
    #--- get item id ---
    top_n_item_id = [this_data[0] for this_data in predict_rating_item_id] # => get item id
    
    #--- Get recommendation item is existing in datasets ---
    recommend_result = item_df[item_df["parent_asin"].isin(top_n_item_id)]
    
    return recommend_result

#--- Get user pervious reviewed items ---
def get_user_reviewed_items(review_df, item_df, user_id, n=8):
    user_review_df = review_df[review_df["user_id"] == user_id]
    user_review_df = user_review_df.sort_values(by="rating", ascending=False)
    user_review_df = user_review_df.iloc[0:n]
    user_review_item_id = list(user_review_df["parent_asin"])
    
    return user_review_item_id, item_df[item_df["parent_asin"].isin(user_review_item_id)]

#--- Show square components for dashboard layout ---
def show_square_components(square_name, value, icon_style="fa-user", square_style="vertical", idname=""):
    if(square_style == "vertical"): #--- for vertical style ---
        sq_comp = html.Div(children=[
                html.Div(children=[
                    html.Div("", className="fa-solid "+icon_style, style=font_40),
                    html.Div(square_name, style=font_16),
                    html.Div(value, style=font_40)
                ], style=sq_dash)
            ], style=sq_dash_frame, className="col-4"
        )
    elif(square_style == "horizontal"): #--- for horizontal style ---
        sq_comp = html.Div(children=[
                html.Div(children="", className="fa-solid "+icon_style+" col-4", style=re_dash_left_frame),
                html.Div(children=[
                    html.Div(value, style=font_40),
                    html.Div(square_name, style=font_20),
                ], className="col-8", style=re_dash_right_frame),
            ], className="row", style=re_dash_frame)
    elif(square_style == "left"): #--- for left menu style ---
        sq_comp = html.Div(children=[
                        html.Div(children="", className="fa-solid "+icon_style+" col-4", style=re_dash_left_small_frame),
                        html.Div(children=[
                            html.Div(value, style=font_15, id=idname),
                            html.Div(square_name, style=font_10),
                        ], className="col-8", style=re_dash_right_small_frame),
                    ], className="row", style=re_dash_frame) 
    return sq_comp

#--- show customer rating bar chart ---
def show_customer_rating_barchart(review_df):
    user_rating_count_df = review_df["rating"].value_counts().reset_index()
    fig = px.bar(user_rating_count_df, x="rating", y="count")
    fig.update_layout(
        height=180,
        margin=dict(
            l=10,
            r=10,
            b=20,
            t=10,
            pad=4
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),  
        xaxis_title=None
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    
    fig = dcc.Graph(figure=fig, id="cs_r_barchart")
    html_code = html.Div(children=[
        html.Div(children=[
            html.Div("Customer Review rating distribution", style=dash_small_title),
            fig
        ], style=msq_dash)
    ], style=msq_dash_frame, className="col-3")
    return html_code

#--- show customer sentiement piechart ---    
def show_customer_sentiment_piechart(review_df):
    user_sentiment_count_df = review_df["user_sentiment"].value_counts().reset_index()

    fig = px.pie(user_sentiment_count_df, values='count', names='user_sentiment', color='user_sentiment', hole=.3) 
    fig.update_layout(
        margin=dict(
            l=10,
            r=10,
            b=30,
            t=10,
            pad=4
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),       
    )   
    fig = dcc.Graph(figure=fig, id="cs_piechart")
    
    html_code = html.Div(children=[
         html.Div(children=[
            html.Div("Overview Customer sentiment", style=dash_small_title),
            fig
         ], style=msq_dash)
    ], style=msq_dash_frame, className="col-3")
    return html_code

#--- show customer sentiemnt barchart ---
def show_customer_sentiment_barchart(review_df, start_date, end_date, min_date, max_date, stylename=msq_dash_frame):
    user_sentiment_count_df = review_df.groupby(by=["year_month", "user_sentiment"]).agg({"user_sentiment" : "count"})
    user_sentiment_count_df.columns = ["count"]
    user_sentiment_count_df = user_sentiment_count_df.reset_index()
    user_sentiment_count_df["filter_year_month"] = pd.to_datetime(user_sentiment_count_df["year_month"]) #==> convert datetime 
    
    #--- handle start date and end date section ---
    start_date = str(start_date)
    end_date = str(end_date)
    list_start_date = start_date.split("-")
    start_date = str(list_start_date[0])+"-"+str(list_start_date[1])+"-01"
    list_end_date = end_date.split("-")
    end_date = str(list_end_date[0])+"-"+str(list_end_date[1])+"-01"
    #--- end handle start date and end date section ---
    
    #--- filter dataframe ---
    user_sentiment_count_df = user_sentiment_count_df[(start_date <= user_sentiment_count_df["filter_year_month"]) & (user_sentiment_count_df["filter_year_month"] <= end_date)]
    
    #--- Show diagram ---
    fig = px.bar(user_sentiment_count_df, x="year_month", y="count", color="user_sentiment", text_auto='.2s', color_discrete_sequence=["red", "green", "blue"]) #, title="Long-Form Input"
    fig.update_layout(
        xaxis_tickangle=-80,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(
            l=0,
            r=10,
            b=30,
            t=10,
            pad=4
        ),
        xaxis_title=None
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(dtick="M1", tickformat='%Y-%m')
    barchart_fig = dcc.Graph(figure=fig, id="cs_barchart")
    
    html_code = html.Div(children=[
        html.Div(children=[
            html.Div("Customer sentiment by month", style=dash_small_title, className="col-4"),
            html.Div(children=[dcc.DatePickerRange(
                id='cs-date-picker-range',
                display_format='YYYY-MM',
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                initial_visible_month=max_date,
                start_date=start_date,
                end_date=end_date,
            )], className="col-8"),
            barchart_fig
        ], style=msq_dash, className="row")
    ], style=stylename, className="col-4")
    return html_code, fig

#--- Show datatable ---
def show_datatable(df, hidden_cols=[], show_cols=[], filter_action="native", header_bg_color="black", header_font_color="white"):
    if(len(hidden_cols) > 0):
        df = df.drop(hidden_cols, axis=1)
    
    if(len(show_cols) > 0):
        df = df[show_cols]
    
    html_code = html.Div(children=[
        dash_table.DataTable(
            id='datatable-interactivity',
            columns=[
                {"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns
            ],
            data=df.to_dict('records'),
            editable=False,
            filter_action=filter_action,
            sort_action="native",
            sort_mode="multi",
            #column_selectable="single",
            row_selectable="multi",
            row_deletable=False,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current= 0,
            page_size= 25,
            style_cell={
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'textAlign': 'left',
                'padding': '5px',
                'maxWidth': 0,
                'fontSize':12, 
                'font-family':'sans-serif',
                'whiteSpace': 'pre-line' #=> auto newline
            },     
            style_header={
                'backgroundColor': header_bg_color,
                'fontWeight': 'bold',
                'color' : header_font_color,
                'fontSize':14, 
                'font-family':'sans-serif'
            },   
        ),
        html.Div(children=[""], id="warning-message"),
        #html.Div(id='datatable-interactivity-container')  
    ], className="col-12")
    
    return html_code
    
#--- Show recommendation item ---
def show_recommendation_item(user_id, user_predict_result, item_df, is_hidden, enduser_version=False):
    global total_recommendation_item, dict_recommendation_item, dict_recommendation_item_click_rate, global_message, log_userfile
    
    if(enduser_version == False):
        html_title = html.H1("Manage user recommended magazines", style=text_align_center)
    else:
        html_title = html.H1("Recommend New Magazine", style=text_align_center)
        
    if(enduser_version == False):
        html_dropdown = dcc.Dropdown(get_user_for_select(review_df), user_id, placeholder="Please select user ID", id='user-dropdown')
    else:
        html_dropdown = html.Div("")
    
    
    html_alert = html.Div(children=[global_message], id="warning-message")
    user_recommendation_html = [html_title,  html_dropdown, html_alert, html.Br(), html.Br(), html.Hr()] #dcc.Location(id="url2", refresh=False)
    
    if(user_id != ""):
        #--- Get user reviewd item ---
        user_review_item_id, user_reviewed_items_df = get_user_reviewed_items(review_df, item_df, user_id, n=show_number_of_user_reviewed_items)
        
        #--- Get model predict recommendation item ---
        user_recommendation_item_df = get_top_n_recommendation_items(user_predict_result, item_df, user_id, n=show_number_recommendation_items)
        
        #--- Remove item in user reviewed before ---
        user_recommendation_item_df = user_recommendation_item_df[~user_recommendation_item_df["parent_asin"].isin(user_review_item_id)]
        
        #--- Show model predict user recommmendation item ---
        user_recommendation_html.append(html.Div("Recommend New Magazines:", style=user_recommendation_title))
        for index, row in user_recommendation_item_df.iterrows():
            this_item_id = row["parent_asin"]
            this_title = row["title"]
            this_large_image_url = row["large_image"]
            
            html_img = html.Div(children=[html.Img(src=this_large_image_url, width="100%")]) #--- Show magazine image ---
            html_title = html.Div(this_title, style=product_title_text)
            
            if(enduser_version == True):
                html_product_name = html.A(children=[html_title, html_img], href="?click="+this_item_id, style=black_font)#--- Show product name ---
                html_buy_product = html.A(children=["[ Buy ]"], href="?buy="+this_item_id, rel=this_item_id, style=bold)#--- Show buy link---
            else:
                html_product_name = html.Div(children=[html_title, html_img], style=black_font) #--- Show product name ---
                
                this_item_click_rate = 0
                if(not dict_recommendation_item_click_rate[this_item_id]):
                    this_item_click_rate = 0
                else:
                    this_item_click_rate = dict_recommendation_item_click_rate[this_item_id]
                
                this_item_total_buy = 0
                if(not dict_recommendation_item_buy[this_item_id]):
                    this_item_total_buy = 0
                else:
                    this_item_total_buy = dict_recommendation_item_buy[this_item_id]
                 
                html_item_click_rate = html.Div("Total click : " + str(this_item_click_rate), style=red_font)
                html_item_total_buy = html.Div("Total Purchase : " + str(this_item_total_buy), style=green_font)
                html_buy_product = html.Div(children=[html_item_click_rate, html_item_total_buy], style=bold)#--- Show buy link---
                
            html_code = html.Div(children=[html_product_name, html_buy_product], className="col-2", style=list_item_frame)
            
            user_recommendation_html.append(html_code)
            
            if(enduser_version == True):
                if(global_message == ""): #--- prevent duplicate count ---
                    #--- count total recommendation item ---
                    total_recommendation_item = total_recommendation_item + 1
                
                    if(this_item_id not in dict_recommendation_item):
                        dict_recommendation_item[this_item_id] = 1
                    else:
                        dict_recommendation_item[this_item_id] = dict_recommendation_item[this_item_id] + 1
                    
                if(this_item_id not in dict_recommendation_item_click_rate):
                    dict_recommendation_item_click_rate[this_item_id] = 0
                
                if(this_item_id not in dict_recommendation_item_buy):
                    dict_recommendation_item_buy[this_item_id] = 0 
                
                #--- Save log section ---
                log_dict["dict_recommendation_item"] = dict_recommendation_item
                log_dict["dict_recommendation_item_click_rate"] = dict_recommendation_item_click_rate
                log_dict["dict_recommendation_item_buy"] = dict_recommendation_item_buy
                log_dict["total_recommendation_item"] = total_recommendation_item
                log_dict["total_click_item"] = total_click_item
                save_log(log_dict, log_userfile)
                #--- end Save log section ---
    
        #--- Show user previous reviewed item ---
        user_recommendation_html.append(html.Hr())
        user_recommendation_html.append(html.Div("Previously reviewed magazines:", style=user_recommendation_title))
        for index, row in user_reviewed_items_df.iterrows():
            this_item_id = row["parent_asin"]
            this_title = row["title"]
            this_large_image_url = row["large_image"]
            
            html_img = html.Div(children=[html.Img(src=this_large_image_url, width="100%")])
            html_product_name = html.Div(this_title) #html.A(this_title, href="#"+this_item_id)
            html_code = html.Div(children=[html_img, html_product_name], className="col-2", style=list_item_frame)
            
            user_recommendation_html.append(html_code)
    
    html_code = user_recommendation_html
    
    return html_code, total_recommendation_item

#--- Member Login section ---
def show_login(show_status, titlename="Member Login"):
    
    #--- Alert message ---
    alert_div = html.Div("", className="col-12", style=red_font, id="alert_message")
    
    #--- Login title ---
    login_title = html.H4(titlename, style=text_align_left, className="col-12")
    
    #--- Username section ---
    username_label = html.Div("User ID: ", className="col-3", style=bold)
    username_input = html.Div(children=[dcc.Input(id="input_username", type="text", placeholder="", debounce=True)], className="col-9", style=margin_top_10)
    
    #--- Password section ---
    password_label = html.Div("Password: ", className="col-3", style=bold)
    password_input = html.Div(children=[dcc.Input(id="input_password", type="password", placeholder="", debounce=True)], className="col-9", style=margin_top_10)
    
    #--- Login button section ---
    login_label = html.Div("", className="col-3")
    login_button = html.Div(children=[html.Button('Login', id='submit-val', n_clicks=0, className="btn btn-primary")], className="col-9", style=margin_top_10)
    
    #--- html login section ---
    html_login = html.Div(children=[login_title, alert_div, username_label, username_input, password_label, password_input, login_label, login_button], className="row", style=show_status)
    
    return html_login

#--- Show left sidebar ---
def show_left_sidebar(stylename):
    left_sidebar = html.Div(
        [
            html.H3("Magazine Recommendation system prototype", className="display-7", style=stylename),
            html.Hr(),
            html.P(
                "Dashboard", className="lead", style=stylename
            ),
            dbc.Nav(
                [
                    dbc.NavLink([
                        html.Div([
                            html.I(className="fa-solid fa-home"),
                            " Home",
                        ]),
                    ], href="/dashboard", active="exact"), #=> for dashboard page 
                    dbc.NavLink([
                        html.Div([
                            html.I(className="fa-solid fa-comments"),
                            " Top Reviews",
                        ]),
                    ], href="/review", active="exact"),  #=> for user review page 
                    dbc.NavLink([
                        html.Div([
                            html.I(className="fa-solid fa-box-archive"),
                            " All Magazines",
                        ]),
                    ], href="/item", active="exact"), #=> for item page    
                    dbc.NavLink([
                        html.Div([
                            html.I(className="fa-solid fa-thumbs-up"),
                            " Manage User Recommendation",
                        ]),
                    ], href="/recommendation", active="exact"), #=> for management recommendation page 
                    dbc.NavLink([
                        html.Div([
                            html.I(className="fa-solid fa-user-tag"),
                            " End user Recommendation",
                        ]),
                    ], href="/login", active="exact", target="_blank", external_link=True), #=> for end-user recommendation page 
                    html.Hr(),
                    dbc.NavLink([
                        html.Div([
                            html.I(className="fa-solid fa-right-from-bracket"),
                            " Logout",
                        ]),
                    ], href="/logout", active="exact"), #=> for logout page 
                    show_square_components("Total recommended Magazines", total_recommendation_item, icon_style="fa-flag", square_style="left", idname="left_total_recommend_item"),     
                    show_square_components("Total Click rate", total_click_item, icon_style="fa-arrow-pointer", square_style="left", idname="left_total_click_rate"),    
                    html.Div(children=[], id="div-update-status"),
                ],
                vertical=True,
                pills=True,
                className="bg-light",
                style=stylename,
            ),
        ],
        style=SIDEBAR_STYLE,
        id="left_sidebar",
    )
    
    return left_sidebar

#------ end All Common function ------

#--- Read JSON ---
item_df = get_item(item_json_file)
review_df, min_date, max_date = get_review(review_json_file)
#--- end Read JSON ---

#--- Load Machine Learning Model ---
user_predict_result, trained_model = load_model(model_fullpath)
#--- end Load Machine Learning Model ---

#--- Load log file---
log_dict = load_log(log_userfile)

if("dict_recommendation_item" in log_dict):
    dict_recommendation_item = log_dict["dict_recommendation_item"]

if("dict_recommendation_item_click_rate" in log_dict):
    dict_recommendation_item_click_rate = log_dict["dict_recommendation_item_click_rate"]

if("dict_recommendation_item_buy" in log_dict):
    dict_recommendation_item_buy = log_dict["dict_recommendation_item_buy"]
    
if("total_recommendation_item" in log_dict):
    total_recommendation_item = log_dict["total_recommendation_item"]

if("total_click_item" in log_dict):  
    total_click_item = log_dict["total_click_item"]
#--- end Load log file ---


#--- Set BOOTSTRAP and font style in dash ---
app = Dash(external_stylesheets=[
    dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME
])
server=app.server #==> this code is used by render server 

#--- Some preload components ---
html_recommendation_item1, _ = show_recommendation_item("", "", "", True)
html_recommendation_item2, _ = show_recommendation_item("", "", "", False)
html_recommendation_section1 = html.Div(children=html_recommendation_item1, hidden=True, className="row", id="recommendation-content-user") #=> pre-load some components
html_recommendation_section2 = html.Div(children=html_recommendation_item2, hidden=True, className="row", id="recommendation-content-admin") #=> pre-load some components
html_cs_barchart, cs_barchart_fig = show_customer_sentiment_barchart(review_df, start_date, max_date, min_date, max_date, msq_dash_frame_hidden)
html_login = show_login(hidden_html)
html_temp = html.Div(children=[], id="dummyhtml")

#--- Create timer for daily update ---
daily_timer = dcc.Interval(id='interval-component', interval=24*60*60*1000, n_intervals=0)

#--- Create Core Layout ---
app.layout = html.Div(children=[dcc.Location(id="url"), #, refresh=False
                                daily_timer,
                                show_left_sidebar(hidden_html), #--- Show Left sidebar ---
                                html.Div(children=[html_recommendation_section1, html_recommendation_section2, html_cs_barchart, html_login, html_temp], style=CONTENT_STYLE, id="main-page-content"),  #--- main page content --- 
                                
])
#--- end Create Core Layout ---

#------ All Callback function ------

#--- Callback for render page ---
@app.callback([Output("main-page-content", "children"), Output("left_sidebar", "children")], [Input("url", "pathname")])
def render_page_content(pathname):
    global total_recommendation_item, total_click_item, dict_recommendation_item, dict_recommendation_item_click_rate, dict_recommendation_item_buy, df_user_id, login_user_id, admin_user_id, global_message, global_pathname
    print("Page path : " + pathname)
    
    global_pathname = pathname
    if pathname == "/": #--- Show admin login page ----
        login_user_id = ""
        global_message = ""
        
        #--- Create Login page ---
        login_title = html.H1("Dashboard", style=text_align_center)
        html_login = show_login({}, "Admin Login")
        
        return html_login, login_title
    
    elif pathname == "/dashboard": #--- Show dashboard page ---
        
        #--- Check amdin login ----
        if(admin_user_id == ""):
            return dcc.Location(pathname="/", id="logout"), ""
        
        df_user_id, total_user = get_total_number_of_user(review_df)
        sq_comb_for_total_user = show_square_components("Total number of Users", total_user, "fa-user")

        total_item = get_total_number_of_item(review_df)
        sq_comb_for_total_item = show_square_components("Total number of Magazines", total_item, "fa-book")        

        total_review = get_total_number_of_review(review_df)
        sq_comb_for_total_reivew = show_square_components("Total number of Reviews", total_review, "fa-comment")     
        
        #--- Show customer sentiment pie chart ---
        cs_piechart = show_customer_sentiment_piechart(review_df)
        
        #--- Show customer sentiment bar chart ---
        html_cs_barchart, cs_barchart_fig = show_customer_sentiment_barchart(review_df, start_date, max_date, min_date, max_date, msq_dash_frame)
        
        #--- Show customer rating bar chart ---
        cs_rate_barchart = show_customer_rating_barchart(review_df)
        
        #--- Show Recommended magazines section ---     
        list_recommendation_item_id = list(dict_recommendation_item.keys())
        recommendation_item_df = item_df[item_df["parent_asin"].isin(list_recommendation_item_id)]
        recommendation_item_df["total_recommend"] = recommendation_item_df["parent_asin"].apply(lambda x : dict_recommendation_item[x])
        recommendation_item_df["total_click"] = recommendation_item_df["parent_asin"].apply(lambda x : dict_recommendation_item_click_rate[x])
        recommendation_item_df = recommendation_item_df.sort_values(by=["total_recommend"], ascending=False)
        
        list_recommendation_item_id = list(dict_recommendation_item_buy.keys())
        recommendation_item_buy_df = item_df[item_df["parent_asin"].isin(list_recommendation_item_id)]["parent_asin"].reset_index()
        recommendation_item_buy_df["total_purchase"] = recommendation_item_buy_df["parent_asin"].apply(lambda x : dict_recommendation_item_buy[x])
        
        recommendation_item_df = pd.merge(left=recommendation_item_df, right=recommendation_item_buy_df, left_on="parent_asin", right_on="parent_asin", how="left")        
        recommendation_item_df["total_purchase"] = recommendation_item_df["total_purchase"].fillna(value=0) #--- fill na to zero ---
        recommendation_item_df["total_purchase"] = recommendation_item_df["total_purchase"].astype(int) #--- convert to integer ---
       
        recommendation_frame_left = html.Div(children=[
            show_square_components("Total recommended Magazines", total_recommendation_item, icon_style="fa-flag", square_style="horizontal", idname=""), 
            show_square_components("Total Click rate", total_click_item, icon_style="fa-arrow-pointer", square_style="horizontal", idname=""), 
        ], className="col-4")
        
        recommendation_frame_recommend_item = html.Div(children=[
            html.Div(children=[
                html.Div("Model Recommend Magazines for user: ", style=dash_small_title),
                show_datatable(recommendation_item_df, hidden_cols=[], show_cols=["parent_asin", "title", "total_recommend", "total_click", "total_purchase"], filter_action="native", header_bg_color="orange", header_font_color="black")
            ], className="row", style=recommend_item_frame),
        ], className="col-5")
        
        recommendation_frame_top_item = html.Div(children=[
            html.Div(children=[
                html.Div("Top "+str(top_n_high_rate_and_review_item) + " high rating and review item: ", style=dash_small_title),
                show_datatable(get_top_high_rate_and_review_item(review_df, item_df, n=top_n_high_rate_and_review_item), hidden_cols=[], show_cols=[], filter_action="native")
            ], className="row", style=high_rate_and_review_item_frame),
        ], className="col-5") 
        #--- end Show Recommended magazines section ---   
       
        return html.Div(children=[sq_comb_for_total_user, sq_comb_for_total_item, sq_comb_for_total_reivew, 
                                  recommendation_frame_left, recommendation_frame_recommend_item, cs_rate_barchart, 
                                  html_cs_barchart, cs_piechart, recommendation_frame_top_item], className="row"), show_left_sidebar(show_html)
        
    elif pathname == "/review": #--- Show review datatable page ---
        show_top_40_review_df = review_df.sort_values(by=["timestamp"], ascending=False)
        show_top_40_review_df = show_top_40_review_df.head(show_top_number_of_review)
        review_datatable = show_datatable(show_top_40_review_df, ["images", "date", "date_year", "date_month", "date_day" , "date_week", "year_month"]) #=> show datatable
        review_title = html.H1("Show the Lastest User Review", style=text_align_center)
        return html.Div(children=[review_title, review_datatable], className="row"), show_left_sidebar(show_html)
        
    elif pathname == "/item": #--- Show item datatable page ---
        show_item_df = item_df
        item_datatable = show_datatable(show_item_df, [], ["parent_asin", "title", "average_rating", "rating_number", "large_image"]) #=> show datatable
        item_title = html.H1("Show Magazines data", style=text_align_center)
        return html.Div(children=[item_title, item_datatable], className="row"), show_left_sidebar(show_html)
   
    elif pathname == "/recommendation": #--- Show personal recommendation page ---
        #login_user_id = ""
        global_message = ""
        html_recommendation_item, _ = show_recommendation_item("", "", "", False, False)
        return html.Div(children=html_recommendation_item, hidden=False, className="row", id="recommendation-content-admin"), show_left_sidebar(show_html)
        
    elif pathname == "/login": #--- Show member login page ---
        login_user_id = ""
        global_message = ""
        
        #--- Create Login page ---
        login_title = html.H1("Login", style=text_align_center)
        html_login = show_login({}, "Member Login")
        
        return html_login, login_title
    elif pathname == "/recommendclient": #--- Manage recommended magazine pages---
        this_user_id = login_user_id
        html_recommendation_item, total_recommendation_item = show_recommendation_item(this_user_id, user_predict_result, item_df, False, True)
        welcome_title = html.Div(children=[html.H1("Welcome: ", style=text_align_left), html.Div(login_user_id, style=font_13)])
        
        return html.Div(children=html_recommendation_item, hidden=False, className="row", id="recommendation-content-user"), welcome_title
    elif pathname == "/logout": #--- Logout section ---
        admin_user_id = ""
        return dcc.Location(pathname="/", id="logout"), ""
        
    #--- For unreached page ---
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    ),show_left_sidebar(show_html)

#--- Callback for recommend new magazines ---
@app.callback(Output('recommendation-content-admin', 'children'), Input('user-dropdown', 'value'))
def update_recommendation_output(value):
    global global_message #login_user_id,  df_user_id, global_pathname, 
    
    this_user_id = value    
    total_recommendation_item = 0
    html_recommendation_item = ""
    
    if(this_user_id != ""):
        global_message = ""
    
    #--- Recommend new magazines to user ---
    html_recommendation_item, total_recommendation_item = show_recommendation_item(this_user_id, user_predict_result, item_df, False, False)
    
    return html_recommendation_item
    

#--- Callback for log customer behavior(click rate, purchases) ---
@app.callback(Output("warning-message", "children"), [Input("url", "href")])
def update_click_item(href):
    global total_click_item, global_href, global_message, log_userfile
    
    message = ""
    if(href != "" and "?" in href and global_href != href): #--- Check url ---        
        this_url, this_parameters = href.split("?")
        this_user_action, this_item_id = this_parameters.split("=") #--- Get url parameters ---
        
        if(this_user_action == "click"): #--- For click rate ---
            message = "User click " + str(this_item_id) + " magazine"
            total_click_item = total_click_item + 1
            if(this_item_id not in dict_recommendation_item_click_rate):
                dict_recommendation_item_click_rate[this_item_id] = 1
            else:
                dict_recommendation_item_click_rate[this_item_id] = dict_recommendation_item_click_rate[this_item_id] + 1
 
        elif(this_user_action == "buy"): #--- For purchases ---
            message = "User buy " + str(this_item_id) + " magazine"
            if(this_item_id not in dict_recommendation_item_buy):
                dict_recommendation_item_buy[this_item_id] = 1
            else:
                dict_recommendation_item_buy[this_item_id] = dict_recommendation_item_buy[this_item_id] + 1  
            

        #--- Save log section ---
        log_dict["dict_recommendation_item"] = dict_recommendation_item
        log_dict["dict_recommendation_item_click_rate"] = dict_recommendation_item_click_rate
        log_dict["dict_recommendation_item_buy"] = dict_recommendation_item_buy
        log_dict["total_recommendation_item"] = total_recommendation_item
        log_dict["total_click_item"] = total_click_item
        save_log(log_dict, log_userfile)
        #--- end Save log section ---
        
        global_href = href
        global_message = message
        
        return dcc.Location(href="/recommendclient?"+this_user_action+"="+this_item_id, id="recommendclient") #--- redirect recommend client page ---
    else:
        global_href = href
        return global_message #--- return message ---

#--- Callback for customer sentiement barchart ---
@app.callback(Output('cs_barchart', 'figure'), Input('cs-date-picker-range', 'start_date'), Input('cs-date-picker-range', 'end_date'))
def update_cs_barchart(start_date, end_date):
    html_cs_barchart, cs_barchart_fig = show_customer_sentiment_barchart(review_df, start_date, end_date, min_date, max_date)
    return cs_barchart_fig

#--- Callback for member login ----
@app.callback(Output('alert_message', 'children'), Input('submit-val', 'n_clicks'), [State('input_username', 'value'), State('input_password', 'value')], prevent_initial_call=True)
def check_member_login(n_clicks, input_username, input_password):
    global login_user_id, admin_user_id, df_user_id, global_message, global_pathname
    global_message = ""
    
    if(n_clicks >= 1):
        if(global_pathname == "/"):
            #--- For Administrator section ---
            if(admin_name == input_username and admin_password == input_password):
                admin_user_id = admin_name
                return dcc.Location(pathname="/dashboard", id="dashboard")
            else:
                return html.Div("Login Failed!")
        else:
            #--- For Customer section ---
            if(df_user_id[df_user_id["user_id"] == input_username].empty == False and input_password == member_default_password):
                login_user_id = input_username
                return dcc.Location(pathname="/recommendclient", id="recommendclient")
            else:
                return html.Div("Login Failed!")

#--- Callback for model and data daily ---
@app.callback(Output('div-update-status', 'children'), Input('interval-component', 'n_intervals'))
def update_model_and_data_daily(n_intervals):
    global user_predict_result, trained_model, item_df, review_df, min_date, max_date
    
    #--- Get today datetime ---
    today_datetime = datetime.datetime.now()
    
    #--- re-load model and data ---
    user_predict_result, trained_model = load_model(model_fullpath)
    item_df = get_item(item_json_file)
    review_df, min_date, max_date = get_review(review_json_file)
    #--- end re-load model and data ---
    
    #--- output html ---
    html_code = html.Div(children=[
        html.Div("Daily Update Model & Data:", style=bold),
        html.Div(today_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    ])
    
    return html_code


#--- end All Callback function ---
    
#--- Start dashboard service ---
if __name__ == "__main__":
    if(start_service_model == "debug"):
        app.run(debug=True) #=> debug model
    elif(start_service_model == "production"):
        app.run_server(port=8888) #=> production model
    