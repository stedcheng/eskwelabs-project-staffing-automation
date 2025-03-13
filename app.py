import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import time, timedelta

st.set_page_config(layout = 'wide')

df = pd.read_csv('Mentor Systems - Mentor Pool Candidate Information v1.2_RedactedInfo.csv')
abbreviations = pd.read_csv('abbreviations.csv')

########## DATASET PREPROCESSING ##########

# make dictionaries
column_name_to_code = abbreviations.set_index('column_name')['code'].to_dict()
title_to_code = abbreviations.set_index('title')['code'].to_dict()

# shorten column names
df.columns = [column_name_to_code[col] if col in column_name_to_code.keys() else col for col in df.columns]

# remove duplicate columns
df = df[list(df.columns[:35]) + list(df.columns[60:])]

# remove non-alphanumeric characters
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9]+', ' ', text)

# replace erroneous values and encode
replace_dict = {'Morning Shift 7am to 12nn Afternoon Shift 1pm to 6pm Evening Shift 6pm to 9pm Not available for the whole day' : 'N',
                'Morning Shift 7am to 12nn Not available for the whole day' : 'N',
                'Afternoon Shift 1pm to 6pm Not available for the whole day' : 'N', 
                'Evening Shift 6pm to 9pm Not available for the whole day' : 'N',
                'Not available for the whole day' : 'N',
                'Morning Shift 7am to 12nn' : 'M',
                'Afternoon Shift 1pm to 6pm' : 'A',
                'Evening Shift 6pm to 9pm' : 'E',
                'Morning Shift 7am to 12nn Afternoon Shift 1pm to 6pm' : 'MA',
                'Morning Shift 7am to 12nn Evening Shift 6pm to 9pm' : 'ME',
                'Afternoon Shift 1pm to 6pm Evening Shift 6pm to 9pm' : 'AE',
                'Morning Shift 7am to 12nn Afternoon Shift 1pm to 6pm Evening Shift 6pm to 9pm' : 'MAE'
                }
for col in list(df.columns[df.columns.str.contains('GMT')]):
    # fill missing values
    df[col] = df[col].fillna('N')
    # remove non-alphanumeric characters and spaces
    df[col] = df[col].apply(clean_text)
    df[col] = df[col].str.strip()
    # replace erroneous values and encode
    df[col] = df[col].replace(replace_dict, regex = True)

# one hot encode
days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
schedule_dict = dict(zip(list(df.columns[df.columns.str.contains('GMT')]), days_of_week))
for key, value in schedule_dict.items():
    df[value + '_morning'] = np.where(df[key].str.contains('M'), 1, 0)
    df[value + '_afternoon'] = np.where(df[key].str.contains('A'), 1, 0)
    df[value + '_evening'] = np.where(df[key].str.contains('E'), 1, 0)

# filter out red flags and inactive members
df = df[df['Red Flags?'] == 'No']
df = df[df['Pool Status'] == 'Active']

# drop unnecessary columns
all_redacted = [col for col in df.columns if (df[col].unique() == np.array(['redacted'])).all()]
df.drop(all_redacted, axis = 1, inplace = True)
df.drop(df.columns[df.columns.str.contains('GMT')], axis = 1, inplace = True)
df.drop(['Timestamp', 'Red Flags?', 'Pool Status'], axis = 1, inplace = True)

#st.write('Dataframe')
#st.write(df)

########## FUNCTIONS ##########

def check_time(df, day, start_time, end_time):

    # preliminary error checking
    if start_time > end_time: 
        raise ValueError('Start time is later than end time')
    if day not in days_of_week:
        raise ValueError('Day not in days of week. Use abbreviations Mon, Tue, Wed, Thu, Fri, Sat, Sun')
    if start_time < time(12, 0, 0) and end_time > time(13, 0, 0):
        raise ValueError('This crosses the lunch hour 1200-1300')
        
    if end_time < time(12, 0, 0): # morning
        time_filter = np.where(df[day + '_morning'] == 1, 1, 0)
    elif start_time > time(13, 0, 0) and end_time < time(18, 0, 0): # afternoon
        time_filter = np.where(df[day + '_afternoon'] == 1, 1, 0)
    elif start_time < time(18, 0, 0) and end_time > time(18, 0, 0): # afternoon and evening
        time_filter = np.where((df[day + '_afternoon'] == 1) & (df[day + '_evening'] == 1), 1, 0)
    elif start_time > time(18, 0, 0): # evening
        time_filter = np.where(df[day + '_evening'] == 1, 1, 0)
    else: # catch-all for errors
        time_filter = np.array([-1] * len(df))

    # filter rows where the person is available
    df_time = df[time_filter == 1]
    #st.write(df_time)
    return df_time

def check_skills(df, skill_list, threshold_list, weight_list):
    # preliminary error checking
    if len(skill_list) != len(threshold_list):
        raise ValueError('Skill and threshold lists are not of equal length')
    if len(skill_list) != len(weight_list):
        raise ValueError('Skill and weight lists are not of equal length')
    if len(threshold_list) != len(weight_list):
        raise ValueError('Threshold and weight lists are not of equal length')

    adj_skill_score_list = []
    skill_list = [title_to_code[skill_list[i]] for i in range(len(skill_list))]
    for i in range(len(skill_list)):
        # if the exam score is below the threshold, then the adjusted score is 0
        # else if the exam score is greater than or equal to the threshold, then the adjusted score is the weight multiplied by
        #    1 + the number of standard deviations above the threshold, which I believe has been standardized to 20
        adj_skill_score = np.where(
            df[skill_list[i]] > threshold_list[i], 
            weight_list[i] * (1 + (df[skill_list[i]] - threshold_list[i])/20), 
            0)
        adj_skill_score_list.append(adj_skill_score)

    # create a dataframe with skills as columns and people identification as index
    df_skills = pd.DataFrame(adj_skill_score_list).T
    df_skills.columns = skill_list
    df_skills.index = df.index

    # calculate total adjusted skill score by summing per row
    df_skills['total_adj_skill_score'] = df_skills.sum(axis = 1)

    # merge this table with the original df 
    df_skills = pd.merge(df, df_skills, left_index = True, right_index = True, suffixes = ['_exam', '_adj'])

    # sort by total adjusted skill score
    df_skills = df_skills.sort_values(by = 'total_adj_skill_score', ascending = False)
    #st.write(df_skills)

    # check if any of the _adj columns are 0
    adj_skill_score_cols = df_skills.columns[df_skills.columns.str.contains('_adj')]
    
    # remove any rows where at least one _adj column is 0
    df_skills = df_skills[~(df_skills[adj_skill_score_cols] == 0).any(axis = 1)]
    if len(df_skills) == 0:
        # st.write('No mentors found.')
        return pd.DataFrame()
    else:
        #st.write(df_skills)
        return df_skills

def check_faci(df, threshold, weight, faci = 'mentor'):
    if 'total_adj_skill_score' not in df.columns:
        return pd.DataFrame()
    elif faci == 'mentor':
            # filter rows with CSAT greater than threshold
            df = df[df['Average Mentor CSAT'] > threshold]
            # mentor score is equal to the adjusted skill score plus the adjusted mentor CSAT score
            df['mentor_score'] = df['total_adj_skill_score'] + weight * (1 + (df['Average Mentor CSAT'] - threshold)/df['Average Mentor CSAT'].std())
            df = df.sort_values(by = 'mentor_score', ascending = False)
    else:
        df = df[(df['Average Instructor CSAT'] > threshold) & (~df['Average Instructor CSAT'].isna())]
        # instructor score is equal to the adjusted skill score plus the adjusted instructor CSAT score
        df['instructor_score'] = df['total_adj_skill_score'] + weight * (1 + (df['Average Instructor CSAT'] - threshold)/df['Average Instructor CSAT'].std())
        df = df.sort_values(by = 'instructor_score', ascending = False)
    #st.write(df)
    return df

def check_all(df, day, start_time, end_time, skill_list, threshold_list, weight_list, threshold, weight):
    # st.write('Matching schedules...')
    df_time = check_time(df, day, start_time, end_time)
    # st.write('Matching skills...')
    df_skills = check_skills(df_time, skill_list, threshold_list, weight_list)
    # st.write('Matching facilitators...')
    df_mentor = check_faci(df_skills, threshold, weight, 'mentor')
    df_instructor = check_faci(df_skills, threshold, weight, 'instructor')
    if 'mentor_score' not in df_mentor.columns:
        return pd.DataFrame(), 0, pd.DataFrame(), 0
    elif 'instructor_score' not in df_instructor.columns:
        exam_cols = list(df_mentor.columns[df_mentor.columns.str.contains('exam')])
        return df[['mentor_score'] + exam_cols + ['Average Mentor CSAT']].head(5), len(df_mentor), pd.DataFrame(), 0
    else:
        df_instructor = df_instructor[~df_instructor['instructor_score'].isna()]
        exam_cols = list(df_mentor.columns[df_mentor.columns.str.contains('exam')])
        return df_mentor[['mentor_score'] + exam_cols + ['Average Mentor CSAT']].head(5), len(df_mentor), \
               df_instructor[['instructor_score'] + exam_cols + ['Average Instructor CSAT']].head(5), len(df_instructor)

########## INPUTS AND DISPLAY ##########

st.title('Eskwelabs Project Staffing Automation')

tab1, tab2 = st.tabs(['Input', 'Variables and Formulas'])

with tab1:
    with st.container():
        col1, col2, col3 = st.columns([1/3, 1/3, 1/3])
        with col1: day = st.selectbox('Day', options = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        with col2: start_time = st.time_input('Start Time', value = time(20, 0, 0), step = timedelta(minutes = 30))
        with col3: end_time = st.time_input('End Time', value = time(21, 30, 0), step = timedelta(minutes = 30))

    skill_list = st.multiselect('Skills Needed', options = abbreviations['title'])
    threshold_list = []
    weight_list = []
    with st.container():
        col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
        with col1: st.write('Topic')
        with col2: st.write('Diagnostic Exam Score Threshold')
        with col3: st.write('Weight')
    for i in range(len(skill_list)):
        with st.container():
            col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
            with col1: st.write(i+1, skill_list[i])
            with col2:
                threshold = st.number_input('Diagnostic Exam Score Threshold', min_value = 0.0, max_value = 100.0, value = 50.0, key = f'threshold{i+1}', label_visibility = 'collapsed')
                threshold_list.append(threshold)
            with col3:
                weight = st.number_input(f'Weight', min_value = 0.0, value = 1/len(skill_list), key = f'weight{i+1}', label_visibility = 'collapsed')
                weight_list.append(weight)

    with st.container():
        col1, col2 = st.columns([0.5, 0.5])
        with col1: csat_threshold = st.number_input('CSAT Threshold', min_value = 0.0, max_value = 10.0, value = 8.0)
        with col2: csat_weight = st.number_input('CSAT Weight', min_value = 0.0, value = 1.0)
        
    button = st.button('Generate Matches!')

    if button:
        top_mentors, num_mentors, top_instructors, num_instructors = check_all(df, day, start_time, end_time, skill_list, threshold_list, weight_list, csat_threshold, csat_weight)

        with st.container():
            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                if num_mentors == 0:
                    st.write('No available mentors')
                elif num_mentors == 1:
                    st.write('1 available mentor')
                    st.write(top_mentors)
                elif num_mentors <= 5:
                    st.write(f'{num_mentors} available mentors')
                    st.write(top_mentors)
                else:
                    st.write(f'{num_mentors} available mentors, top 5 shown')
                    st.write(top_mentors)
            with col2:
                if num_instructors == 0:
                    st.write('No available instructors')
                elif num_instructors == 1:
                    st.write('1 available instructor')
                    st.write(top_instructors)
                elif num_instructors <= 5:
                    st.write(f'{num_instructors} available instructors')
                    st.write(top_instructors)
                else:
                    st.write(f'{num_instructors} available instructors, top 5 shown')
                    st.write(top_instructors)

with tab2:
    st.latex(r'''
\begin{aligned}
    n_f &= \text{number of faciliators (from dataset)} \\
    n_s &= \text{number of skills needed (from dataset)} \\
    e_{fs} &= \text{exam score for facilitator $f$ and skill $s$ (from dataset)} \\
    m_f &= \text{mentor CSAT for facilitator $f$ (from dataset)} \\
    i_f &= \text{instructor CSAT for faciliator $f$ (from dataset)} \\
    \sigma_m &= \text{standard deviation of mentor CSAT (calculated from dataset)} \\
    \sigma_i &= \text{standard deviation of instructor CSAT (calculated from dataset)} \\
    t_s &= \text{threshold exam score for skill $s$ (user input, default 50)} \\
    w_s &= \text{weight for skill $s$ (user input, default $\dfrac{1}{n_s}$)} \\
    t' &= \text{threshold for customer satisfaction (CSAT) (user input, default 8)} \\
    w' &= \text{weight for customer satisfaction (CSAT) (user input, default 1)} \\
    a_{fs} &= \text{adjusted skill score for facilitator $f$ and skill $s$} =
        \left\{\begin{aligned}
            &w_s\left(1+\frac{e_{fs}-t_s}{20}\right) &&\text{if }e_{fs}>t_s \\
            &0 &&\text{otherwise}
        \end{aligned}\right. \\
    A_f &= \text{total adjusted skill score for facilitator $f$} = \sum_{s=1}^{n_s}a_{fs} \\
    M_f &= \text{mentor score for facilitator $f$} =
        \left\{\begin{aligned}
            &A_f + w'\left(1+\frac{m_f-t'}{\sigma_m}\right) &&\text{if }m_f>t' \\
            &0 &&\text{otherwise}
        \end{aligned}\right. \\
    I_f &= \text{instructor score for facilitator $f$} =
        \left\{\begin{aligned}
            &A_f + w'\left(1+\frac{i_f-t'}{\sigma_i}\right) &&\text{if }i_f>t' \\
            &0 &&\text{otherwise}
        \end{aligned}\right. \\
\end{aligned}''')
