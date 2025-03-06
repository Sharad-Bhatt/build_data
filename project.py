import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import random
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

# Database functions
def create_database():
    conn = sqlite3.connect('student_performance.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            grade REAL,
            attendance REAL,
            study_hours INTEGER,
            is_sample INTEGER DEFAULT 1,  -- 1 for sample data, 0 for user-entered
            last_updated TIMESTAMP
        )
    ''')
    conn.commit()
    return conn, cursor

def generate_sample_data(n_students=20):
    names = ['John', 'Emma', 'Michael', 'Sarah', 'David', 'Lisa', 'James', 'Emily', 'Robert', 'Sophie',
             'William', 'Olivia', 'Thomas', 'Ava', 'Joseph', 'Mia', 'Charles', 'Amelia', 'Henry', 'Isabella']
    data = []
    for i in range(1, n_students + 1):
        student = {
            'name': f"{random.choice(names)} {chr(65 + i % 26)}",
            'age': random.randint(15, 18),
            'grade': round(random.uniform(60, 100), 1),
            'attendance': round(random.uniform(70, 100), 1),
            'study_hours': random.randint(5, 20),
            'is_sample': 1,
            'last_updated': datetime.now()
        }
        data.append(student)
    return data

def insert_data(conn, cursor, data):
    cursor.executemany('''
        INSERT INTO students 
        (name, age, grade, attendance, study_hours, is_sample, last_updated)
        VALUES (:name, :age, :grade, :attendance, :study_hours, :is_sample, :last_updated)
    ''', data)
    conn.commit()

def initialize_database(conn, cursor):
    cursor.execute("SELECT COUNT(*) FROM students WHERE is_sample = 1")
    count = cursor.fetchone()[0]
    if count == 0:
        initial_data = generate_sample_data(20)
        insert_data(conn, cursor, initial_data)
        st.success("Initialized database with 20 sample student records!")

def get_data(conn, is_sample=1):
    df = pd.read_sql_query(f"SELECT * FROM students WHERE is_sample = {is_sample}", conn)
    return df

# Prediction function
def predict_grade(study_hours, attendance, df):
    X = df[['study_hours', 'attendance']]
    y = df['grade']
    model = LinearRegression()
    model.fit(X, y)
    pred_df = pd.DataFrame([[study_hours, attendance]], columns=['study_hours', 'attendance'])
    prediction = model.predict(pred_df)
    return max(60, min(100, prediction[0]))

# Streamlit dashboard
def main():
    st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
    
    conn, cursor = create_database()
    initialize_database(conn, cursor)
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Add Student", "Prediction"])
    
    # Sample data for training
    sample_df = get_data(conn, is_sample=1)
    
    with tab1:  # Dashboard
        st.title("Student Performance Dashboard")
        st.markdown("Analysis of student academic performance")
        
        if st.button("Generate Fresh Sample Data"):
            with st.spinner("Generating new sample data..."):
                cursor.execute("DELETE FROM students WHERE is_sample = 1")
                sample_data = generate_sample_data(20)
                insert_data(conn, cursor, sample_data)
                st.success("Generated fresh sample data successfully!")
            sample_df = get_data(conn, is_sample=1)
        
        # Filters
        st.sidebar.header("Filters")
        min_grade = st.sidebar.slider("Minimum Grade", 60, 100, 60)
        age_filter = st.sidebar.multiselect("Age Filter", options=sorted(sample_df['age'].unique()), 
                                          default=sorted(sample_df['age'].unique()))
        
        filtered_df = sample_df[(sample_df['grade'] >= min_grade) & (sample_df['age'].isin(age_filter))]
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Grade", f"{filtered_df['grade'].mean():.1f}")
        with col2:
            st.metric("Average Attendance", f"{filtered_df['attendance'].mean():.1f}%")
        with col3:
            st.metric("Average Study Hours", f"{filtered_df['study_hours'].mean():.1f}")
        
        # Visualizations
        st.header("Data Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.scatter(filtered_df, x="study_hours", y="grade", 
                            color="age", hover_data=['name'],
                            title="Study Hours vs Grade")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            avg_grade_by_age = filtered_df.groupby('age')['grade'].mean().reset_index()
            fig2 = px.bar(avg_grade_by_age, x="age", y="grade",
                         title="Average Grade by Age")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Data table
        st.header("Sample Student Data")
        st.dataframe(filtered_df[['name', 'age', 'grade', 'attendance', 'study_hours']],
                    use_container_width=True)
        
        # Correlation
        correlation = filtered_df['study_hours'].corr(filtered_df['grade'])
        st.markdown(f"**Correlation between study hours and grades:** {correlation:.3f}")

    with tab2:  # Add Student
        st.header("Add New Student Data")
        with st.form("student_form"):
            name = st.text_input("Student Name")
            age = st.number_input("Age", min_value=15, max_value=18, step=1)
            grade = st.number_input("Grade", min_value=60.0, max_value=100.0, step=0.1)
            attendance = st.number_input("Attendance %", min_value=70.0, max_value=100.0, step=0.1)
            study_hours = st.number_input("Study Hours per Week", min_value=5, max_value=20, step=1)
            
            submitted = st.form_submit_button("Add Student")
            if submitted:
                new_student = [{
                    'name': name,
                    'age': age,
                    'grade': grade,
                    'attendance': attendance,
                    'study_hours': study_hours,
                    'is_sample': 0,
                    'last_updated': datetime.now()
                }]
                insert_data(conn, cursor, new_student)
                st.success(f"Added {name} to the database!")

        # Display user-entered students
        user_df = get_data(conn, is_sample=0)
        if not user_df.empty:
            st.header("User-Entered Students")
            st.dataframe(user_df[['name', 'age', 'grade', 'attendance', 'study_hours']],
                        use_container_width=True)

    with tab3:  # Prediction
        st.header("Grade Prediction")
        pred_study_hours = st.slider("Study Hours", 5, 20, 10)
        pred_attendance = st.slider("Attendance %", 70, 100, 85)
        if st.button("Predict Grade"):
            predicted_grade = predict_grade(pred_study_hours, pred_attendance, sample_df)
            st.write(f"Predicted grade for {pred_study_hours} study hours and {pred_attendance}% attendance: {predicted_grade:.1f}")
        
        # Show top performers from sample data
        st.header("Top 5 Sample Performers")
        top_students = sample_df.nlargest(5, 'grade')[['name', 'grade', 'study_hours']]
        st.table(top_students)

    conn.close()

if __name__ == "__main__":
    main()