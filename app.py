import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, date, time, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit_sortables import sort_items

st.set_page_config(layout = 'wide')

df = pd.read_csv('Mentor Systems - Mentor Pool Candidate Information v1.2_RedactedInfo.csv')
abbreviations = pd.read_csv('abbreviations.csv')

### SAMPLE JSON
# {"n_events": 2, "priority_list": [1, 2], "priority_type": "Absolute", "date_matrix": [["2025-03-19"], ["2025-03-19"]], "day_matrix": [["Wednesday"], ["Wednesday"]], "start_time_matrix": [["20:00:00"], ["20:00:00"]], "end_time_matrix": [["21:30:00"], ["21:30:00"]], "schedule_ok_list": [true, true], "skill_matrix": [["Data Visualization in Python"], ["Web Development"]], "threshold_matrix": [[50.0], [50.0]], "weight_matrix": [[1], [1]], "csat_threshold_list": [8.0, 8.0], "csat_weight_list": [1.0, 1.0], "n_mentors_dict": {"0": 5, "1": 5}, "n_instructors_dict": {"0": 2, "1": 2}}

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
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
schedule_dict = dict(zip(list(df.columns[df.columns.str.contains('GMT')]), days_of_week))
for key, value in schedule_dict.items():
    df[value + '_morning'] = np.where(df[key].str.contains('M'), 1, 0)
    df[value + '_afternoon'] = np.where(df[key].str.contains('A'), 1, 0)
    df[value + '_evening'] = np.where(df[key].str.contains('E'), 1, 0)

# filter out red flags and inactive members
len_with_red_inactive = len(df)
df = df[df['Red Flags?'] == 'No']
df = df[df['Pool Status'] == 'Active']
len_no_red_inactive = len(df)
# st.write(f'{len_with_red_inactive} facilitators, minus {len_with_red_inactive - len_no_red_inactive} red flags/inactive pool status = {len_no_red_inactive} facilitators')

# drop unnecessary columns
all_redacted = [col for col in df.columns if (df[col].unique() == np.array(['redacted'])).all()]
df.drop(all_redacted, axis = 1, inplace = True)
df.drop(df.columns[df.columns.str.contains('GMT')], axis = 1, inplace = True)
df.drop(['Timestamp', 'Red Flags?', 'Pool Status'], axis = 1, inplace = True)

# calculate population standard deviations of mentor and instructor CSATs
s_m = df['Average Mentor CSAT'].std()
s_i = df['Average Instructor CSAT'].std()

# st.write('Dataframe')
# st.write(df)

########## IMPLEMENTING FUNCTIONS ##########

def check_schedule(df, day, start_time, end_time):

    # preliminary error checking
    if start_time > end_time: 
        raise ValueError('Start time is later than end time')
    if start_time < time(12, 0, 0) and end_time > time(13, 0, 0):
        raise ValueError('This crosses the lunch hour 1200-1300')
        
    if end_time <= time(12, 0, 0): # morning
        schedule_filter = np.where(df[day + '_morning'] == 1, 1, 0)
    elif start_time >= time(13, 0, 0) and end_time <= time(18, 0, 0): # afternoon
        schedule_filter = np.where(df[day + '_afternoon'] == 1, 1, 0)
    elif start_time < time(18, 0, 0) and end_time > time(18, 0, 0): # afternoon and evening
        schedule_filter = np.where((df[day + '_afternoon'] == 1) & (df[day + '_evening'] == 1), 1, 0)
    elif start_time >= time(18, 0, 0): # evening
        schedule_filter = np.where(df[day + '_evening'] == 1, 1, 0)
    else: # catch-all for errors
        schedule_filter = np.array([-1] * len(df))

    # filter rows where the person is available
    df_schedule = df[schedule_filter == 1]
    #st.write(df_schedule)
    return df_schedule

def check_many_schedules(df, day_list, start_time_list, end_time_list):
    schedules = list(zip(day_list, start_time_list, end_time_list))
    df_schedule = df
    for day, start_time, end_time in schedules:
        old_len = len(df_schedule)
        df_schedule = check_schedule(df_schedule, day, start_time, end_time)
        new_len = len(df_schedule)
        # st.write(f'{day}, {start_time}, {end_time}: number of facilitators reduced from {old_len} to {new_len}')
    return df_schedule

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

    # mark facilitators that are qualified in all skills (no 0 score in any _adj column)
    adj_skill_score_cols = df_skills.columns[df_skills.columns.str.contains('_adj')]
    df_skills['skill_qualified'] = np.where(
        (df_skills[adj_skill_score_cols] == 0).any(axis = 1), 0, 1
    )

    # sort by total adjusted skill score
    df_skills = df_skills.sort_values(by = ['skill_qualified', 'total_adj_skill_score'], ascending = [False, False])
    
    # st.write(df_skills)
    return df_skills

def check_faci(df, threshold, weight, faci = 'mentor'):
    if 'total_adj_skill_score' not in df.columns:
        return pd.DataFrame()
    elif faci == 'mentor':
        df['csat_qualified'] = np.where(df['Average Mentor CSAT'] > threshold, 1, 0)
        # mentor score is equal to the adjusted skill score plus the adjusted mentor CSAT score
        df['score'] = df['total_adj_skill_score'] + np.where(df['csat_qualified'] == 1,
                                                             weight * (1 + (df['Average Mentor CSAT'] - threshold)/s_m), 0)
        
        df = df.sort_values(by = ['skill_qualified', 'csat_qualified', 'score'], ascending = [False, False, False])
    else:
        df['csat_qualified'] = np.where((df['Average Instructor CSAT'] > threshold) & (~df['Average Instructor CSAT'].isna()), 1, 0)
        # instructor score is equal to the adjusted skill score plus the adjusted instructor CSAT score
        df['score'] = df['total_adj_skill_score'] + np.where(df['csat_qualified'] == 1,
                                                             weight * (1 + (df['Average Instructor CSAT'] - threshold)/s_i), 0)
        df = df.sort_values(by = ['skill_qualified', 'csat_qualified', 'score'], ascending = [False, False, False])
    #st.write(df)
    return df

def check_all(df, day_list, start_time_list, end_time_list, skill_list, threshold_list, weight_list, threshold, weight, detailed):
    # st.write('Matching schedules...')
    df_schedule = check_many_schedules(df, day_list, start_time_list, end_time_list)
    # st.write(df_schedule, len(df_schedule))
    # st.write('Matching skills...')
    df_skills = check_skills(df_schedule, skill_list, threshold_list, weight_list)
    # st.write(df_skills, len(df_skills))
    # st.write('Matching facilitators...')
    df_mentor = check_faci(df_skills, threshold, weight, 'mentor')
    df_instructor = check_faci(df_skills, threshold, weight, 'instructor')
    exam_cols = list(df_mentor.columns[df_mentor.columns.str.contains('exam')])
    if detailed:
        return df_mentor[['skill_qualified', 'csat_qualified', 'score'] + exam_cols + ['Average Mentor CSAT']], \
               df_instructor[['skill_qualified', 'csat_qualified', 'score'] + exam_cols + ['Average Instructor CSAT']]
    else:
        return df_mentor[['skill_qualified', 'csat_qualified', 'score']], df_instructor[['skill_qualified', 'csat_qualified', 'score']]

########## FORMATTING FUNCTIONS ##########
    
# Function to serialize datetime objects to strings
def custom_serializer(obj):
    if isinstance(obj, (date, time)):
        return obj.isoformat()  # Converts to a string like '2025-03-19' or '20:00:00'
    raise TypeError("Type not serializable")
# Function to deserialize strings back into datetime objects
def custom_deserializer(dct):
    for key, value in dct.items():
        if isinstance(value, list):
            dct[key] = [
                datetime.fromisoformat(item).date() if 'date' in key and isinstance(item, str) else
                datetime.fromisoformat(item).time() if 'time' in key and isinstance(item, str) else item
                for item in value
            ]
    return dct

# convert date, start time, and end time matrices to ISO format to be compatible with JSON when copying
def datetime_to_string(matrix):
    return [[item.isoformat() if isinstance(item, (date, time)) \
             else item for item in sublist] for sublist in matrix]
# do the reverse after pasting
def string_to_date(matrix):
    return [[datetime.strptime(item, '%Y-%m-%d').date() for item in sublist] for sublist in matrix]
def string_to_time(matrix):
    return [[datetime.strptime(item, '%H:%M:%S').time() for item in sublist] for sublist in matrix]

########## INPUTS AND DISPLAY ##########

st.title('Eskwelabs Project Staffing Automation')

tab1, tab2 = st.tabs(['Input', 'Variables and Formulas'])

with tab1:

    save_checkbox = st.checkbox('Do you have existing input?')
    if save_checkbox:
        st.subheader('With Previous Input')
        save_str = st.text_input('Paste your previous input here, then press Enter:')
        # st.write(save_str)
        try:
            save_dict = json.loads(save_str, object_hook = custom_deserializer)
            # st.write(save_dict)
            n_events, priority_list, priority_type, date_matrix, day_matrix, start_time_matrix, end_time_matrix, \
                      schedule_ok_list, skill_matrix, threshold_matrix, weight_matrix, \
                      csat_threshold_list, csat_weight_list, n_mentors_dict, n_instructors_dict = \
                      save_dict['n_events'], save_dict['priority_list'], save_dict['priority_type'], save_dict['date_matrix'], save_dict['day_matrix'], \
                      save_dict['start_time_matrix'], save_dict['end_time_matrix'], save_dict['schedule_ok_list'], \
                      save_dict['skill_matrix'], save_dict['threshold_matrix'], save_dict['weight_matrix'], \
                      save_dict['csat_threshold_list'], save_dict['csat_weight_list'], save_dict['n_mentors_dict'], save_dict['n_instructors_dict']
            # convert back to proper format
            date_matrix = string_to_date(date_matrix)
            start_time_matrix = string_to_time(start_time_matrix)
            end_time_matrix = string_to_time(end_time_matrix)
            n_mentors_dict = {int(key) : value for key, value in n_mentors_dict.items()}
            n_instructors_dict = {int(key) : value for key, value in n_instructors_dict.items()}
        except:
            st.write('An error occurred. Please check what you pasted. If the error persists, please manually input again.')

    else:
        st.subheader('Without Previous Input')
        with st.container():
            col1, col2, col3 = st.columns([1/3, 1/3, 1/3])
            with col1: n_events = st.number_input('Number of Events', min_value = 1, value = 1)
            with col2:
                st.markdown("<p style='margin-bottom: 5px; font-size: 14px;'>Priority</p>", unsafe_allow_html=True)
                priority_list = [str(i) for i in range(1, n_events + 1)]
                priority_list = sort_items(priority_list)
                priority_list = [int(i) for i in priority_list]
            with col3:
                priority_type = st.radio(label = 'Priority Type', options = ['Absolute', 'Balanced'], index = 0,
                                         help = '''Absolute priority means that all mentors and instructors for the event with the highest priority will be filled first
    before moving to the events with lower priority. Balanced priority means that the slots for mentors and instructors are filled one at a time.''')
               
        date_matrix = []
        day_matrix = []
        start_time_matrix = []
        end_time_matrix = []
        schedule_ok_list = []
        skill_matrix = []
        threshold_matrix = []
        weight_matrix = []
        csat_threshold_list = []
        csat_weight_list = []
        n_mentors_dict = {}
        n_instructors_dict = {}

        for i in range(n_events):
            st.header(f'Event {i+1}')

            ##### SCHEDULES #####
            st.subheader('Schedules')
            date_list = []
            day_list = []
            start_time_list = []
            end_time_list = []
            n_schedules = st.number_input('Number of Schedules', min_value = 1, value = 1, key = f'n_schedules{i+1}')
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([0.1, 0.25, 0.25, 0.2, 0.2])
                with col1: st.write('No.')
                with col2: st.write('Date')
                with col3: st.write('Day of the Week')
                with col4: st.write('Start Time')
                with col5: st.write('End Time')
            schedule_ok = True
            for j in range(n_schedules):
                col1, col2, col3, col4, col5 = st.columns([0.1, 0.25, 0.25, 0.2, 0.2])
                with col1:
                    st.write(f'{j+1}')
                with col2:
                    date_input = st.date_input('Date', value = 'today', key = f'date{i+1},{j+1}', label_visibility = 'collapsed')
                    date_list.append(date_input) # weird variable naming to avoid overshadowing the word 'date' as datatype
                with col3:
                    day = date_input.strftime('%A')
                    st.write(day)
                    day_list.append(day)
                with col4:
                    start_time = st.time_input('Start Time', value = time(20, 0, 0), step = timedelta(minutes = 30), key = f'start_time{i+1},{j+1}', label_visibility = 'collapsed')
                    start_time_list.append(start_time)
                with col5:
                    end_time = st.time_input('End Time', value = time(21, 30, 0), step = timedelta(minutes = 30), key = f'end_time{i+1},{j+1}', label_visibility = 'collapsed')
                    end_time_list.append(end_time)
                if start_time > end_time:
                    st.write(f'Oops! The end time of schedule {j+1} is later than its start time.')
                    schedule_ok = False
                if date_input < pd.Timestamp.today().date():
                    st.write(f'Oops! The date of schedule {j+1} is in the past.')
                    schedule_ok = False
            date_matrix.append(date_list)
            day_matrix.append(day_list)
            start_time_matrix.append(start_time_list)
            end_time_matrix.append(end_time_list)
            schedule_ok_list.append(schedule_ok)

            ##### SKILLS #####
            st.subheader('Skills')
            skill_list = st.multiselect('Skills Needed', options = abbreviations['title'], key = f'skill_list{i+1}')
            threshold_list = []
            weight_list = []
            if len(skill_list) >= 1:
                with st.container():
                    col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
                    with col1: st.write('Topic and Abbreviation')
                    with col2: st.write('Diagnostic Exam Score Threshold')
                    with col3: st.write('Weight')
            for j in range(len(skill_list)):
                with st.container():
                    col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
                    with col1: st.write(f'{j+1}. {skill_list[j]} ({title_to_code[skill_list[j]]})')
                    with col2:
                        threshold = st.number_input('Diagnostic Exam Score Threshold', min_value = 0.0, max_value = 100.0, value = 50.0,
                                                    key = f'threshold{i+1},{j+1}', label_visibility = 'collapsed')
                        threshold_list.append(threshold)
                    with col3:
                        weight = st.number_input(f'Weight', min_value = 0.0, value = 1/len(skill_list), key = f'weight{i+1},{j+1}', label_visibility = 'collapsed')
                        weight_list.append(weight)
            skill_matrix.append(skill_list)
            threshold_matrix.append(threshold_list)
            weight_matrix.append(weight_list)

            ##### CSAT / N_Facis #####
            with st.container():
                col1, col2 = st.columns([0.5, 0.5])
                with col1: st.subheader('Customer Satisfaction')
                with col2: st.subheader('Number of Facilitators')
                col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
                with col1:
                    csat_threshold = st.number_input('CSAT Threshold', min_value = 0.0, max_value = 10.0, value = 8.0, key = f'csat_threshold{i+1}')
                    csat_threshold_list.append(csat_threshold)
                with col2:
                    csat_weight = st.number_input('CSAT Weight', min_value = 0.0, value = 1.0, key = f'csat_weight{i+1}')
                    csat_weight_list.append(csat_weight)
                with col3:
                    n_mentors = st.number_input('Number of Mentors', min_value = 0, value = 5, key = f'n_mentors{i+1}')
                    n_mentors_dict[i] = n_mentors
                with col4:
                    n_instructors = st.number_input('Number of Instructors', min_value = 0, value = 2, key = f'n_instructors{i+1}')
                    n_instructors_dict[i] = n_instructors

            st.divider()

        # Convert date and time to ISO format to be compatible with JSON
        date_matrix_mod = datetime_to_string(date_matrix)
        start_time_matrix_mod = datetime_to_string(start_time_matrix)
        end_time_matrix_mod = datetime_to_string(end_time_matrix)

        variables = { # note double quotes
            'n_events' : n_events,
            'priority_list' : priority_list,
            'priority_type' : priority_type,
            'date_matrix' : date_matrix_mod,
            'day_matrix' : day_matrix,
            'start_time_matrix' : start_time_matrix_mod,
            'end_time_matrix' : end_time_matrix_mod,
            'schedule_ok_list' : schedule_ok_list,
            'skill_matrix' : skill_matrix,
            'threshold_matrix' : threshold_matrix,
            'weight_matrix' : weight_matrix,
            'csat_threshold_list' : csat_threshold_list,
            'csat_weight_list' : csat_weight_list,
            'n_mentors_dict' : n_mentors_dict,
            'n_instructors_dict' : n_instructors_dict
        }
        copy_variables = json.dumps(variables, default = custom_serializer)
        copy_variables = copy_variables.replace("'", '"') # replace single with double quotes
        # st.write(copy_variables)

        st.header('Copying Input')
        if all(schedule_ok_list) and all(bool(skill_list) for skill_list in skill_matrix):
            copy_input = st.checkbox('Copy input', help = '''If you check this, a box below will come out.
You can copy your current input so that you can continue next time from where you stopped.''')
            if copy_input:
                st.code(copy_variables)
        else: st.write('Please check the conflicting schedule or the skills needed.')

    ##### MATCHING #####
    st.header('Matching Facilitators to Classes')
    if save_checkbox == True or (all(schedule_ok_list) and all(bool(skill_list) for skill_list in skill_matrix)):
        with st.container():
            col1, col2, col3 = st.columns([1/3, 1/3, 1/3])
            with col1: button = st.button('Generate Matches!')
            with col2: detailed = st.checkbox('Display detailed results', help = '''If you check this, the diagnostic exam scores for the relevant skills
and the average mentor and instructor CSAT will be shown in the tables''')
            with col3: initial = st.checkbox('Display initial possibilities', help = '''If you check this, the initial (unfiltered) ranked list of mentors and instructors will be shown,
without taking into account overlaps''')
    else:
        button = False
        st.write('Please check the conflicting schedule or the skills needed.')
        # error_df = pd.DataFrame([schedule_ok_list, skill_matrix]).T
        # error_df.columns = ['Schedules', 'Skills Needed']
        # st.write(error_df)

    

    if button:
        ranked_mentors_dict = {}
        ranked_instructors_dict = {}
        permanent_ranked_mentors_dict = {}
        permanent_ranked_instructors_dict = {}
        for i in range(n_events):
            ranked_mentors, ranked_instructors = check_all(df, day_matrix[i], start_time_matrix[i], end_time_matrix[i],
                                                     skill_matrix[i], threshold_matrix[i], weight_matrix[i],
                                                     csat_threshold_list[i], csat_weight_list[i], detailed = detailed)

            ranked_mentors_dict[i] = ranked_mentors
            ranked_instructors_dict[i] = ranked_instructors
            permanent_ranked_mentors_dict[i] = ranked_mentors
            permanent_ranked_instructors_dict[i] = ranked_instructors

            # fix formatting
            ranked_mentors['score'] = ranked_mentors['score'].round(2)
            ranked_instructors['score'] = ranked_instructors['score'].round(2)
            for col in ranked_mentors.columns[ranked_mentors.columns.str.contains('exam')]:
                ranked_mentors[col] = ranked_mentors[col].round(0)
            for col in ranked_instructors.columns[ranked_instructors.columns.str.contains('exam')]:
                ranked_instructors[col] = ranked_instructors[col].round(0)

        # print initial results of facilitators
        if initial:
            st.header('Initial Facilitator Assignments')
            for i in range(n_events):
                with st.container():
                    st.subheader(f'Initial Facilitators for Event {i+1}')
                    col1, col2 = st.columns([0.5, 0.5])    
                    with col1:
                        st.write('Mentors')
                        st.write(ranked_mentors_dict[i])
                    with col2:
                        st.write('Instructors')
                        st.write(ranked_instructors_dict[i])
                st.divider()

        mentor_assignments_dict = {i : [] for i in range(n_events)}
        instructor_assignments_dict = {i : [] for i in range(n_events)}
        assigned_facilitators = []
        
        if priority_type == 'Absolute':

            # assign instructors
            for i in range(n_events): # i = priority level
                # define event number, number of instructors, and ranked df of instructors
                event = priority_list[i]-1
                n_instructors = n_instructors_dict[event]
                ranked_instructors = ranked_instructors_dict[event]
                # filter out the instructors in the assigned instructors, then assign it to a dictionary
                ranked_instructors = ranked_instructors[~ranked_instructors.index.isin(assigned_facilitators)]
                # assign instructors
                instructor_assignments_dict[event] += list(ranked_instructors.index[:n_instructors])
                assigned_facilitators += list(instructor_assignments_dict[event])
                
            # assign mentors
            for i in range(n_events):
                # define event number, number of mentors, and ranked df of mentors
                event = priority_list[i]-1
                n_mentors = n_mentors_dict[event]
                ranked_mentors = ranked_mentors_dict[event]
                # filter out the mentors in the assigned mentors, then assign it to a dictionary
                ranked_mentors = ranked_mentors[~ranked_mentors.index.isin(assigned_facilitators)]
                # assign mentors
                mentor_assignments_dict[event] += list(ranked_mentors.index[:n_mentors])
                assigned_facilitators += list(mentor_assignments_dict[event])
                
        else:

            # define max number of instructors across all events
            max_n_instructors = max(list(n_instructors_dict.values()))
            for i in range(max_n_instructors):
                for j in range(n_events): # j = priority level
                    event = priority_list[j]-1
                    n_instructors = n_instructors_dict[event]
                    if i < n_instructors:
                        ranked_instructors = ranked_instructors_dict[event]
                        ranked_instructors = ranked_instructors[~ranked_instructors.index.isin(assigned_facilitators)]
                        first_instructor = ranked_instructors.index[0]
                        instructor_assignments_dict[event].append(first_instructor)
                        assigned_facilitators.append(first_instructor)
                        ranked_instructors_dict[event] = ranked_instructors

            # define max number of mentors across all events
            max_n_mentors = max(list(n_mentors_dict.values()))
            for i in range(max_n_mentors):
                for j in range(n_events): # j = priority level
                    event = priority_list[j]-1
                    n_mentors = n_mentors_dict[event]
                    if i < n_mentors:
                        ranked_mentors = ranked_mentors_dict[event]
                        ranked_mentors = ranked_mentors[~ranked_mentors.index.isin(assigned_facilitators)]
                        first_mentor = ranked_mentors.index[0]
                        mentor_assignments_dict[event].append(first_mentor)
                        assigned_facilitators.append(first_mentor)
                        ranked_mentors_dict[event] = ranked_mentors

        selected_mentors_dict = {}
        selected_instructors_dict = {}
        # selected mentors and instructors for a particular event
        selected_mentors_event_dict = {i : [] for i in range(n_events)}
        selected_instructors_event_dict = {i : [] for i in range(n_events)}
        
        st.header('Final Facilitator Assignments')
        for i in range(n_events): 
            with st.container():
                st.subheader(f'Facilitators for Event {i+1}')
                col1, col2 = st.columns([0.5, 0.5])
                event = priority_list[i]-1
                with col1:
                    selected_mentors = permanent_ranked_mentors_dict[i].loc[mentor_assignments_dict[i]]
                    n_qualified_mentors = len(selected_mentors[selected_mentors['skill_qualified'] * selected_mentors['csat_qualified'] == 1])
                    n_mentors = n_mentors_dict[event]
                    st.write('Mentors')
                    if n_qualified_mentors == n_mentors:
                        st.write(f'All {n_mentors} qualified')
                    else:
                        st.write(f'{n_qualified_mentors} qualified, {n_mentors - n_qualified_mentors} unqualified')
                    st.write(selected_mentors)
                    selected_mentors_dict[i] = selected_mentors
                    selected_mentors_event_dict[i] += list(selected_mentors.index)
                    
                with col2:
                    selected_instructors = permanent_ranked_instructors_dict[i].loc[instructor_assignments_dict[i]]
                    n_qualified_instructors = len(selected_instructors[selected_instructors['skill_qualified'] * selected_instructors['csat_qualified'] == 1])
                    n_instructors = n_instructors_dict[event]
                    st.write('Instructors')
                    if n_qualified_instructors == n_instructors:
                        st.write(f'All {n_instructors} qualified')
                    else:
                        st.write(f'{n_qualified_instructors} qualified, {n_instructors - n_qualified_instructors} unqualified')
                    st.write(selected_instructors)
                    selected_instructors_dict[i] = selected_instructors
                    selected_instructors_event_dict[i] += list(selected_instructors.index)

                st.divider()

        st.header('Summary of Assignments')
        
        # print summary of results
        selected_mentors_event_dict = {index : event_number + 1 for event_number, index_list in selected_mentors_event_dict.items() for index in index_list}
        selected_mentors_df = pd.DataFrame(selected_mentors_event_dict.items(), columns = ['faci_index', 'event'])
        selected_mentors_df['role'] = np.array(['mentor'] * len(selected_mentors_df))

        selected_instructors_event_dict = {index : event_number + 1 for event_number, index_list in selected_instructors_event_dict.items() for index in index_list}
        selected_instructors_df = pd.DataFrame(selected_instructors_event_dict.items(), columns = ['faci_index', 'event'])
        selected_instructors_df['role'] = np.array(['instructor'] * len(selected_instructors_df)) 

        selected_mentors_qualified_df = pd.concat([selected_mentors_dict[i][['skill_qualified', 'csat_qualified']] for i in range(n_events)])
        selected_instructors_qualified_df = pd.concat([selected_instructors_dict[i][['skill_qualified', 'csat_qualified']] for i in range(n_events)])
        selected_facilitators_qualified_df = pd.concat([selected_mentors_qualified_df, selected_instructors_qualified_df])
        
        selected_facilitators_df = pd.concat([selected_mentors_df, selected_instructors_df], ignore_index = True).sort_values(by = 'faci_index', ascending = True).reset_index().drop(columns = ['index'])
        selected_facilitators_df = pd.merge(selected_facilitators_df, selected_facilitators_qualified_df, left_on = 'faci_index', right_index = True, how = 'left')
        st.write(selected_facilitators_df)
        
        csv = selected_facilitators_df.to_csv(index = False)
        st.download_button(label = 'Download as CSV', data = csv,
                           help = 'Download selected facilitators data (table above) as CSV', file_name = 'selected_facilitators.csv')
                
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
