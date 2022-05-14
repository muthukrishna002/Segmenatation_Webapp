import streamlit as st
import display

st.image('Images/head.gif')
dataset=st.selectbox(' Select Data Class:',("LUNG","LIVER","RETINA","BRAIN"))

st.sidebar.markdown("<h1 style='text-align: left; color: green;'>SRI RAMAKRISHNA ENGINEERING COLLEGE,<br> COIMBATORE</h1>", 
                unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align: left;'>TEAM EXTREMIST:</h1>", 
                unsafe_allow_html=True)
st.sidebar.markdown('''
                    <ol>
                      <li>INDRA KUMAR K</li>
                      <li>HARI KRISHNA M</li>
                      <li>SREE HARAN A</li>
                      <li>RISHIBALAJI RAJASEKAR</li>
                      <li>AISHWARYA LAXMI G</li>
                      <li>SUBASINI R</li>
                      <li>ARIVALAN C P</li>
                    </ol> ''',
                unsafe_allow_html=True)

if dataset=='LUNG':
    st.markdown("<h1 style='text-align: left; color: green;'>"+dataset+"</h1>", 
                unsafe_allow_html=True)
    img = st.file_uploader("Choose a "+dataset+" image file", type=["jpg","png"])
    if img is not None:
        display.disp_png_jpg(img,dataset)
        
elif dataset=='LIVER':
    st.markdown("<h1 style='text-align: left; color: green;'>"+dataset+"</h1>", 
                unsafe_allow_html=True)
    img=st.text_input("Enter location for LIVER file(eg:liver.nii)")
    store=st.text_input("Enter location to save:")
    if img and store:
        x=st.button("Start-Segmentation")
        if x:
            display.disp_nii(img,store)
            x=False
            
elif dataset=='RETINA':
    st.markdown("<h1 style='text-align: left; color: green;'>"+dataset+"</h1>", 
                unsafe_allow_html=True)
    img = st.file_uploader("Choose a "+dataset+" image file", type=["jpg","png"])
    if img is not None:
        display.disp_png_jpg(img,dataset)   
        
else :
    st.markdown("<h1 style='text-align: left; color: green;'>"+dataset+"</h1>", 
                unsafe_allow_html=True)
    img = st.file_uploader("Choose a "+dataset+" image file", type=["jpg","png"])
    if img is not None:
        display.disp_png_jpg(img,dataset)
        
st.markdown("<style> footer {visibility: hidden;}</style>" , unsafe_allow_html=True) 
