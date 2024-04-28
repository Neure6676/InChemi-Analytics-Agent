import streamlit as st
import pandas as pd

from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv

load_dotenv()

st.title('InChemi Analytics Agent')
st.header('for Molecular Data Science 🧪')

st.write("Hi there, 👋 this is your InChemi AI Agent to help you explore the world of molecular data science.")
st.write(" ")

if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}

def clicked(button):
    st.session_state.clicked[button]= True
st.button("Let's get started", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        llm = OpenAI(temperature = 0)

        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True, agent_executor_kwargs={"handle_parsing_errors": True})

        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first few rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            return

        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y =[user_question_variable])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable} in text description")
            st.write(summary_statistics)
            return
        
        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return

        #Main

        st.header('Exploratory data analysis')
        st.subheader('General information about the dataset')

        function_agent()

        st.subheader('Visualization')
        user_question_variable = st.text_input('What variable are you interested in')
        if user_question_variable is not None and user_question_variable !="":
            function_question_variable()

            st.subheader('Further study')

        if user_question_variable:
            user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                function_question_dataframe()
            if user_question_dataframe in ("no", "No"):
                st.write("")

