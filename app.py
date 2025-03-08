#import the libraries
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


# Function to get response from LLAMA 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    # Calling LLama2 model
    # TheBloke/Llama-2-7B-Chat-GGML
    # 'models/llama-2-7b-chat.ggmlv3.q8_0.bin'
    llm = CTransformers(model = 'TheBloke/Llama-2-7B-Chat-GGML', model_file='llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type = 'llama',
                        config = {'max_new_tokens': 256,
                                  'temperature': 0.01})
    # Temperature - A randomness measure for the LLM model. Here, using low temperature -> low randomness -> more predictable outputs

    # Prompt Template
    template = """
            Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words.
            """

    prompt = PromptTemplate(input_variables = ["blog_style", "input_text", 'no_words'],
                            template = template)
    
    # Generate the response from the LLama2 model
    response = llm(prompt.format(blog_style = blog_style, input_text = input_text, no_words = no_words))
    print(response)
    return response

#UI Components

st.set_page_config(page_title="Generate Blogs", 
                    page_icon='🤖', 
                    layout='centered', 
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs with AI🤖")

input_text = st.text_input("Enter the Blog Topic in Brief")

# Creating two more columns for additional fields

col1, col2 = st.columns([5,5]) #provinding width of the columns

with col1:
    no_words = st.text_input('Number of words') # Choose the number of words to generate the blog
with col2:
    blog_style = st.selectbox('Blog Target Audience', ('Researchers', 'Data Scientists', "General"), index = 0)

# Button to start generation process
submit = st.button("Generate the content")

# Final Response
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))

