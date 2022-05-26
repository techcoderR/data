import requests
import io
import streamlit as st
import numpy as np
import pandas as pd


st.header('Crop Predicton App')


with st.form("my_form"):
    st.write("Fill the Form")
    Area = st.number_input('Enter the Area')
    st.write('The Input Area', Area)
    option = st.selectbox(
     'Select your Crop?',
     ('Wheat', 'Bajara', 'Chilli','Rice','Jower','Peas','Sugarcane','Groundnut'))

    st.write('You selected:', option)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Area",Area , "Crop", option)


def BajraPred(Area):
    print("h3")
    url_bajara=("https://raw.githubusercontent.com/techcoderR/data/main/data2.csv")
    download = requests.get(url_bajara).content
    data = pd.read_csv(io.StringIO(download.decode('utf-8')))
    X=data['Production'].values
    Y=data['Area'].values
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    m = len(X)
    num = 0
    den = 0
    for i in range(m):
        num += (X[i] - mean_x) * (Y[i] - mean_y)
        den += (X[i] - mean_x) ** 2
    b1 = num / den
    b0 = mean_y - (b1* mean_x)
    y=b1*Area
    return y



def wheat(Area):
        url_wheat=("https://raw.githubusercontent.com/techcoderR/data/main/data1.csv")
        download = requests.get(url_wheat).content
        data = pd.read_csv(io.StringIO(download.decode('utf-8')))
        df=data[data.columns[-2:]]
        X=data['Production'].values
        Y=data['Area'].values
        mean_x = np.mean(X)
        mean_y = np.mean(Y)
        m = len(X)
        num = 0
        den = 0
        for i in range(m):
            num += (X[i] - mean_x) * (Y[i] - mean_y)
            den += (X[i] - mean_x) ** 2
        b1 = num / den
        b0 = mean_y - (b1* mean_x)
        y=b1*Area;
        return y;

def ChiliPred(Area):
    url_chilles=("https://raw.githubusercontent.com/techcoderR/data/main/data3.csv")
    download = requests.get(url_chilles).content
    data = pd.read_csv(io.StringIO(download.decode('utf-8')))
    X=data['Production'].values
    Y=data['Area'].values
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    m = len(X)
    num = 0
    den = 0
    for i in range(m):
        num += (X[i] - mean_x) * (Y[i] - mean_y)
        den += (X[i] - mean_x) ** 2
    b1 = num / den
    b0 = mean_y - (b1* mean_x)
    y=b1*Area
    return y

def RicePred(Area):
    url_rice=("https://raw.githubusercontent.com/techcoderR/data/main/data4.csv")
    download = requests.get(url_rice).content
    data = pd.read_csv(io.StringIO(download.decode('utf-8')))
    X=data['Production'].values
    Y=data['Area'].values
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    m = len(X)
    num = 0
    den = 0
    for i in range(m):
        num += (X[i] - mean_x) * (Y[i] - mean_y)
        den += (X[i] - mean_x) ** 2
    b1 = num / den
    b0 = mean_y - (b1* mean_x)
    y=b1*Area
    return y
def JowarPred(Area):
    url_rice=("https://raw.githubusercontent.com/techcoderR/data/main/data5.csv")
    download = requests.get(url_rice).content
    data = pd.read_csv(io.StringIO(download.decode('utf-8')))
    X=data['Production'].values
    Y=data['Area'].values
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    m = len(X)
    num = 0
    den = 0
    for i in range(m):
        num += (X[i] - mean_x) * (Y[i] - mean_y)
        den += (X[i] - mean_x) ** 2
    b1 = num / den
    b0 = mean_y - (b1* mean_x)
    y=b1*Area
    return y
def GroundnutPred(Area):
    url_rice=("https://raw.githubusercontent.com/techcoderR/data/main/data6.csv")
    download = requests.get(url_rice).content
    data = pd.read_csv(io.StringIO(download.decode('utf-8')))
    X=data['Production'].values
    Y=data['Area'].values
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    m = len(X)
    num = 0
    den = 0
    for i in range(m):
        num += (X[i] - mean_x) * (Y[i] - mean_y)
        den += (X[i] - mean_x) ** 2
    b1 = num / den
    b0 = mean_y - (b1* mean_x)
    y=b1*Area
    return y
def SugarcanePred(Area):
    url_rice=("https://raw.githubusercontent.com/techcoderR/data/main/data8.csv")
    download = requests.get(url_rice).content
    data = pd.read_csv(io.StringIO(download.decode('utf-8')))
    X=data['Production'].values
    Y=data['Area'].values
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    m = len(X)
    num = 0
    den = 0
    for i in range(m):
        num += (X[i] - mean_x) * (Y[i] - mean_y)
        den += (X[i] - mean_x) ** 2
    b1 = num / den
    b0 = mean_y - (b1* mean_x)
    y=b1*Area
    return y
def PeasPred(Area):
    url_rice=("https://raw.githubusercontent.com/techcoderR/data/main/data7.csv")
    download = requests.get(url_rice).content
    data = pd.read_csv(io.StringIO(download.decode('utf-8')))
    X=data['Production'].values
    Y=data['Area'].values
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    m = len(X)
    num = 0
    den = 0
    for i in range(m):
        num += (X[i] - mean_x) * (Y[i] - mean_y)
        den += (X[i] - mean_x) ** 2
    b1 = num / den
    b0 = mean_y - (b1* mean_x)
    y=b1*Area
    return y

st.write("Outside the form")
if option=="Wheat":
    st.write("Your Selected crop",option)
    PW=wheat(Area)
    st.write("Your production",PW,"Ton")
elif option=="Bajara":
    PB=BajraPred(Area)
    st.write("Your Selected crop",option)
    st.write("Your Production",PB,"Ton")
elif option=="Chilli" :
    PC=ChiliPred(Area)
    st.write("Your Selected crop",option,)
    st.write("Your Production is", PC,"Ton")
elif option=="Rice":
    PR=RicePred(Area)
    st.write("Your Selected crop",option,)
    st.write("Your Production is", PR,"Ton")
elif option=="Jower":
   PJ=JowarPred(Area)
   st.write("Your Selected crop",option,)
   st.write("Your Production is", PJ,"Ton")
elif option=="Groundnut":
    PG=GroundnutPred(Area)
    st.write("Your Selected crop",option,)
    st.write("Your Production is", PG,"Ton")
elif option=="Peas":
    PP=PeasPred(Area)
    st.write("Your Selected crop",option,)
    st.write("Your Production is", PP,"Ton")
elif option=="Sugarcane":
    PS=SugarcanePred(Area)
    st.write("Your Selected crop",option,)
    st.write("Your Production is", PS,"Ton")






# df_train=df[:30]
# #this present first 30 sample
# df_test=df[-30:]
# #this present last 30 sample

# df_ytrain=df[:30]
# #this present first 30 sample


# df_ytest=df[-30:]
# #this present last 30 sample
# # print(df_ytest)

# # print(df.columns)

# # model=linear_model.LinearRegression()
# # model.fit(df_train,df_ytrain)
# # df_predict=model.predict(df_test)
# # plt.xlabel('Area')
# # plt.ylabel('Production')
# # print("Weights",model.coef_)
# # mean_squared_error(df_ytest,df_predict)
# #plt.scatter(df_test,df_ytest)
# #plt.plot(df_ytest,df_predict)
# plt.show()





