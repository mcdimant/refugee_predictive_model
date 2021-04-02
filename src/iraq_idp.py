import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import geopy.distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

#read in and clean conflict data 
conflict_df = pd.read_csv('../data/conflict_data_irq.csv')
conflict_df.drop(0, inplace=True)
conflict_df['date_start'] = pd.to_datetime(conflict_df['date_start'])
conflict_df['date_end'] = pd.to_datetime(conflict_df['date_start'])

#creating dictionary to standardize governorate names across datasets
governorate_dict = {'Al Anbār province':'Anbar', 'Nīnawá province':'Ninewa',
'Baghdād province':'Baghdad','Dahūk province':'Dahuk', 'Diyālá province':'Diyala','Kirkūk province':'Kirkuk',
'Şalāḩ ad Dīn province':'Salah al-Din', 'Karbalā’ province':'Kerbala', 'An Najaf province':'Najaf',
'Bābil province':'Babylon','Arbīl province':'Erbil','Al Başrah  province':'Basrah','Dhī Qār province':'Thi-Qar',
'Al Muthanná province':'Muthanna','As Sulaymānīyah province':'Sulaymaniyah', 'Maysān  province':'Missan',
'Al Qādisīyah province':'Qadissiya', 'Wāsiţ province':'Wassit'}

district_dict = {'Abū Ghurayb district':'Abu Ghraib', 'Al Ba‘āj district':"Al-Ba'aj",
'Al Qā’im district':"Al-Ka'im", 'Al-Faris district':'Al-Fares', 'Hīt district':'Heet',
'Qaḑā’ ‘Ānah':'Ana', 'Qaḑā’ ad Dawr':'Al-Daur', 'Qaḑā’ al Fallūjah':'Falluja', 'Qaḑā’ al Ḩaḑr':'Hatra',
'Qaḑā’ al Ḩamdānīyah':'Al-Hamdaniya', 'Qaḑā’ al Khāliş':'Al-Khalis', 'Qaḑā’ al Maḩmūdīyah':'Mahmoudiya',
'Qaḑā’ al Mawşil':'Mosul', 'Qaḑā’ al Miqdādīyah':'Al-Muqdadiya', 'Qaḑā’ ar Ramādī':'Ramadi',
'Qaḑā’ ar Ruţbah':'Al-Rutba', 'Qaḑā’ ash Shaykhān':'Al-Shikhan', 'Qaḑā’ Balad':'Balad',
'Qaḑā’ Bayjī':'Baiji', 'Qaḑā’ Haditha':'Haditha', 'Qaḑā’ Khānaqīn':'Khanaqin', 'Qaḑā’ Kifrī':'Kifri',
'Qaḑā’ Sāmarrā':'Samarra', 'Qaḑā’ Sharqāţ':'Al-Shirqat', 'Qaḑā’ Tall ‘Afar':'Telafar',
'Qaḑā’ Tikrīt':'Tikrit', 'Qaḑā’ Zākhū':'Zakho', 'Sinjār district':'Sinjar', 'Tallkayf district':'Tilkaif',
'Tooz district':'Tuz Khurmatu', 'Zakho district':'Zakho', "Al ‘Amādīyah":'Amedi', "Al ‘Amādīyah district":'Amedi',
"Al ‘Azīzīyah district":'Al-Azezia', "Al Madā’in district":'Al-Midaina', 'Al Majar al Kabīr district':'Al-Mejar Al-Kabir',
'Al Maymūnah district':'Al-Maimouna', 'Al-Shirqat district':'Al-Shirqat', 'Baladrūz district':'Baladrooz',
'Dāqūq district':'Daquq', 'Kuwaysinjaq district':'Koisnjaq', 'Mergasur District':'Mergasur',
'Paynjuwayn district':'Penjwin', "Qaḑā al ‘Amārah":'Amara', 'Qaḑā al Chibāyish':'Al-Chibayish',
'Qaḑā Karbalā’':'Kerbala', "Qaḑā’ ‘Alī al Gharbī":'Ali Al-Gharbi', 'Qaḑā’ ad Dīwānīyah':'Diwaniya',
'Qaḑā’ Ain Al Tamur':'Ain Al-Tamur', 'Qaḑā’ al Başrah':'Basrah', 'Qaḑā’ al Hāshimīyah':'Hashimiya',
'Qaḑā’ al Ḩawījah':'Al-Hawiga', 'Qaḑā’ al Ḩayy':'Al-Hai', 'Qaḑā’ al Ḩillah':'Hilla',
'Qaḑā’ al Hindīyah':'Al-Hindiya', 'Qaḑā’ al Kūfah':'Kufa', 'Qaḑā’ al Kūt':'Kut',
'Qaḑā’ al Maḩāwīl':'Al-Mahawil', 'Qaḑā’ al Manādhirah':'Al-Manathera', 'Qaḑā’ al Musayyib':'Al-Musayab',
'Qaḑā’ al Qurnah':'Al-Qurna', 'Qaḑā’ an Najaf':'Najaf', 'Qaḑā’ an Nāşirīyah':'Nassriya',
'Qaḑā’ Arbīl':'Erbil', 'Qaḑā’ as Samāwah':'Al-Samawa', 'Qaḑā’ aş Şuwayrah':'Al-Suwaira',
'Qaḑā’ ash Shāmīyah':'Al-Shamiya', 'Qaḑā’ ash Shaţrah':'Al-Shatra', 'Qaḑā’ az Zubayr':'Al-Zubair',
"Qaḑā’ Ba‘qūbah":"Ba'quba", 'Qaḑā’ Chamchamal':'Chamchamal', 'Qaḑā’ Chomān':'Choman',
'Qaḑā’ Dahūk':'Dahuk', 'Qaḑā’ Dibis':'Dabes', 'Qaḑā’ Ḩadīthah':'Haditha', 'Qaḑā’ Ḩalabchah':'Halabja',
'Qaḑā’ Kalār':'Kalar', 'Qaḑā’ Kirkūk':'Kirkuk', 'Qaḑā’ Makhmūr':'Makhmur', 'Qaḑā’ Miqdādīyah':'Al-Muqdadiya',
'Qaḑā’ Pishdar':'Pshdar', 'Qaḑā’ Rāniyah':'Rania', 'Qaḑā’ Shahrbāzār':'Sharbazher', 'Qaḑā’ Shaqlāwah':'Shaqlawa',
'Qaḑā’ Sulaymānīyah':'Sulaymaniya', 'Qaḑā’ Sūq ash Shuyūkh':'Suq Al-Shoyokh', 'Qaḑā’ Zākhū':'Zakho',
'Soran district':'Soran'}

conflict_df.replace(to_replace=governorate_dict, inplace=True)
conflict_df.replace(to_replace=district_dict, inplace=True)

#filtering down conflict data so it only includes events in districts where displacement occurred 
conflict_df = conflict_df[conflict_df['adm_2'].isin(list(district_dict.values()))]

#reads in data on outflow of refugees, concatenates into dataframe
outflow_filepaths = [s for s in listdir("../data/out/")]
out_df = pd.concat((pd.read_csv("../data/out/"+s) for s in outflow_filepaths), ignore_index=True) 
out_df.dropna(how='any', inplace=True)

#cleanup of data types 
out_df['date'] = pd.to_datetime(out_df['date'])
out_df.rename(columns={'Location Name':'Location_name'}, inplace=True)
out_df['Location ID'] = out_df['Location ID'].astype(int)
out_df['Place ID'] = out_df['Place ID'].astype(int)
out_df['Families'] = out_df['Families'].astype(int)
out_df['Individuals'] = out_df['Individuals'].astype(int)
out_df.rename(columns={'Place ID':'Place_ID'}, inplace=True)

#renaming columns so it's clear that refugees are flowing OUT TO these governorates 
out_df.rename(columns={'Anbar':'to_Anbar', 'Babylon':'to_Babylon', 'Baghdad':'to_Baghdad', 
        'Basrah':'to_Basrah', 'Dahuk':'to_Dahuk','Diyala':'to_Diyala', 'Erbil':'to_Erbil',
        'Kerbala':'to_Kerbala', 'Kirkuk':'to_Kirkuk', 'Missan':'to_Missan', 'Muthanna':'to_Muthana',              
    'Najaf':'to_Najaf', 'Ninewa':'to_Ninewa', 'Qadissiya':'to_Qadissiya', 
    'Salahal Din':'to_Salahal Din', 'Sulaymaniyah':'to_Sulaymaniyah', 'Thi Qar':'to_Thi Qar',
       'Wassit':'to_Wassit'}, inplace=True)

#making clear that these features represent the living situation of refugees who have fled their homes
out_df.rename(columns={'Camp':'out_Camp', 'Hostfamilies':'out_Hostfamilies', 
        'Hotel Motel':'out_Hotel Motel', 'Informalsettlements':'out_Informalsettlements',
       'Own Property':'out_Own Property', 'Other':'out_Other', 
       'Religiousbuilding':'out_Religiousbuilding', 'Rented pre Apr 2019':'out_Rented pre Apr 2019',
       'Rented Habitable': 'out_Rented Habitable', 'Rented Uninhabitable':'out_Rented Uninhabitable', 
       'Schoolbuilding':'out_Schoolbuilding','Unfinishedbuilding':'out_Unfinishedbuilding', 
        'Unknownsheltertype':'out_Unknownsheltertype'}, inplace=True)

#reads in data on returning refugees, concatenates into large dataframe 
returnee_filepaths = [f for f in listdir("../data/inflow/")]
ret_df = pd.concat((pd.read_csv("../data/inflow/"+f)
                  for f in returnee_filepaths), ignore_index=True)

#renaming features to clarify that refugees are returning to this type of shelter
ret_df.rename(columns={'Camp':'ret_camp',
       'Habitual Pre_31_October2018':'ret_Habitual Pre_31_October2018',
      'Habitual Residence (Habitable)':'ret_Habitual Residence (Habitable)',
       'Habitual Residence (Uninhabitable)':'ret_Habitual Residence (Uninhabitable)', 
       'Host_families':'ret_Host_families', 'Hotel_Motel':'ret_Hotel_Motel',
       'Informal_settlements':'ret_Informal_settlements', 
       'Other':'ret_Other', 'Religious_building':'ret_Religious_building', 
       'Rented_houses':'ret_Rented_houses', 'School_building':'ret_School_building', 
       'Unfinished_Abandoned_building':'ret_Unfinished_Abandoned_building',
       'Unknown_shelter_type':'ret_Unknown_shelter_type'}, inplace=True)

#renaming features to clarify that refugees are RETURNING FROM these governorates 
ret_df.rename(columns={'Anbar':'from_Anbar', 'Babylon':'from_Babylon',
       'Baghdad':'from_Baghdad', 'Basrah':'from_Basrah', 'Dahuk':'from_Dahuk',
       'Diyala':'from_Diyala', 'Erbil':'from_Erbil', 'Kerbala':'from_Kerbala', 
        'Kirkuk':'from_Kirkuk','Missan':'from_Missan', 'Muthanna':'from_Muthanna',
        'Najaf':'from_Najaf', 'Ninewa':'from_Ninewa', 'Qadissiya':'from_Qadissiya', 
        'Salahal Din':'from_Salahal Din'}, inplace=True)

#creating dictionary to standardize naming convention for wave of displacement
displacement_dict = {'Pre June14 Period of displacement':'disp_preJun14',
                      'June July14 Period of displacement':'disp_JunJuly14',
                      'August14 Period of displacement':'disp_Aug14',
                      'Post September 14 Period of displacement':'disp_postSep14',
                      'Post April15 Period of displacement':'disp_postApr15',
                      'Post March 16 Period of displacement':'disp_postMar16',
                      'Post 17 October 16 Period of displacement': 'disp_post17Oct16',
                      'July 17 Period of displacement':'disp_Jul17',
                      'Jan19':'disp_Jan19'}

#applying column name standardization
ret_df.rename(columns=displacement_dict, inplace=True)
out_df.rename(columns=displacement_dict, inplace=True)

ret_df.dropna(how='all', inplace=True)
ret_df['date'] = pd.to_datetime(ret_df['date'])

def trim_all_columns(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.applymap(trim_strings)

ret_df = trim_all_columns(ret_df)
out_df = trim_all_columns(out_df)


#Returnee data is cumulative, so it's necessary to get the delta by date (i.e., on 1OCT2015, 25
#households were [freshly] displaced, rather than as of 1OCT2015 100 households have been displaced )
ret_df.sort_values(['Location ID', 'date'], inplace=True)
ret_df.dropna(how='any', inplace=True)
ret_df['ret_delta'] = ret_df.groupby(['Location ID'])['Returnee Families'].transform(lambda x: x.diff()) 

ret_df.sort_values(['Location ID', 'date'], inplace=True)
ret_df.reset_index(inplace=True)
ret_df.drop(columns='index', axis=1, inplace=True)

#should be functionized 
for i in range(len(ret_df)):
    if np.isnan(ret_df.at[i, 'ret_delta']):
        ret_df.at[i, 'ret_delta'] = ret_df.at[i, 'Returnee Families']
    else:
        None

#merging returnee and outflow data
master_df = ret_df.merge(out_df, how='outer', on=['Location ID', 'date', 'Governorate',
                                                 'District', 'Place_ID', 'Location_name',
                                                 'disp_preJun14', 'disp_JunJuly14', 'disp_Aug14', 
                                                'disp_postSep14', 'disp_postApr15',
                                                'disp_postMar16', 'disp_post17Oct16', 
                                                  'disp_Jul17', 'disp_Jan19'])

master_df.rename(columns={'Families':'outflow', 'ret_delta':'inflow'}, inplace=True)

#dropping irrelevant columns and those that would imply leakage from the test data
master_df.drop(columns=['Unnamed: 0_x', 'Latitude_x', 'Longitude_x', 'Latitude_y', 'Longitude_y',
                       'Unnamed: 0_y', 'Arabic_name', 'Governorate',
                        'Location_name', 'Returnee Individuals', 'Returnee Families',
                        'Arabic Name', 'from_Anbar', 'from_Babylon', 'from_Baghdad',
       'from_Basrah', 'from_Dahuk', 'from_Diyala', 'from_Erbil',
       'from_Kerbala', 'from_Kirkuk', 'from_Missan', 'from_Muthanna',
       'from_Najaf', 'from_Ninewa', 'from_Qadissiya', 'from_Salahal Din',
       'Sulaymaniyah', 'Thi Qar', 'Wassit', 'ret_camp', 'Individuals',
       'ret_Habitual Pre_31_October2018', 'ret_Habitual Residence (Habitable)',
       'ret_Habitual Residence (Uninhabitable)', 'ret_Host_families',
       'ret_Hotel_Motel', 'ret_Informal_settlements', 'ret_Other',
       'ret_Religious_building', 'ret_Rented_houses', 'ret_School_building', 'Place_ID',
       'ret_Unfinished_Abandoned_building', 'ret_Unknown_shelter_type', 'Location ID'], inplace=True)

#making date ordinal so it can fit into the X dataset for ML algorithms 
master_df['date'] = master_df['date'].apply(lambda x: x.toordinal())
master_df.fillna(0, inplace=True)

#Groups conflict data by district and date and sums the best estimate of deaths (for that district)
#and date. Returns dataframe. 
g_conflict = pd.DataFrame({'count' : conflict_df.groupby( [ 'adm_2', 'date_start'])
                           ['best'].sum()}).reset_index()

#Makes date_start ordinal for algorithmic processing, renames columns 
g_conflict['date_start'] = g_conflict['date_start'].apply(lambda x: x.toordinal())
g_conflict.rename(columns={'adm_2':'District', 'date_start':'date', 'count':'death_est'}, 
                 inplace=True)

#filters down g_conflict so it matches timeframe of master_df
g_conflict = g_conflict[g_conflict['date'] > min(master_df['date'])]

#feature engeineering, adds estimated number of deaths for matching district and date
master_df = master_df.merge(g_conflict, how='left', on=['date', 'District'])

#Final bit of cleanup, dropping 'District' because it is non-numeric
master_df = master_df.fillna(0)
master_df.drop(columns=['District'], inplace=True)

#setting up X and y for supervised learning algorithms 
X = master_df.drop(columns=['inflow'])
y = master_df['inflow']
X_train, X_test, y_train, y_test = train_test_split(X, y)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

print("RF score:", rf.score(X_test, y_test))

#hyper parameter tuning: Builds tree num vs. score chart
num_trees = range(5, 50, 5)
accuracies = []
for n in num_trees:
    tot = 0
    for i in range(5):
        rf = RandomForestRegressor(n_estimators=n)
        rf.fit(X_train, y_train)
        tot += rf.score(X_test, y_test)
    accuracies.append(tot / 5)
    
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(num_trees, accuracies)
ax.set_xlabel("Number of Trees")
ax.set_ylabel("RF Regressor Score")
ax.set_title('RF Regressor Score vs. Num Trees')
plt.savefig('../images/numtrees_vs_score.png')

#hyper parameter tuning: builds feature num vs. score chart
num_features = range(1, len(X.columns) + 1)
accuracies = []
for n in num_features:
    tot = 0
    for i in range(5):
        rf = RandomForestRegressor(max_features=n)
        rf.fit(X_train, y_train)
        tot += rf.score(X_test, y_test)
    accuracies.append(tot / 5)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(num_features, accuracies)
ax.set_xlabel("Number of Features")
ax.set_ylabel("Score")
ax.set_title('Score vs. Num Features')