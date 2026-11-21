"""
Project 2: Student Academic Performance Analytics
Industry: Education / EdTech
Dataset: 02_student_performance.csv
Description: Inspects, cleans, transforms, and analyzes student attendance,
             study habits, and assessment scores using Python, NumPy, and Pandas.
"""

import numpy as np
import pandas as pd


def load_and_inspect(filepath: str) -> pd.DataFrame:
    """Task 1: Load the dataset and inspect dimensions, data types, and summary stats."""
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("1. DATASET INSPECTION")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn Data Types:")
    print(df.dtypes)
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nDescriptive Statistics:")
    print(df.describe(include="all"))
    return df


def check_quality_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Tasks 2, 3, 4: Check missing values, duplicates, score ranges (0-100), and impute attendance."""
    print("\n" + "=" * 60)
    print("2–4. DATA QUALITY CHECKS & CLEANING")
    print("=" * 60)

    # Task 2: Missing values and duplicate Student_ID values
    print("Missing Values per Column:")
    print(df.isnull().sum())

    dup_ids = df["Student_ID"].duplicated().sum()
    dup_rows = df.duplicated().sum()
    print(f"\nDuplicate Student_ID count: {dup_ids}")
    print(f"Exact duplicate rows count: {dup_rows}")

    # Task 3: Handle missing Attendance_Percent using median imputation
    # Justification: Median (78.2%) is robust against skewed attendance values and preserves sample size.
    df_clean = df.copy()
    attendance_median = df_clean["Attendance_Percent"].median()
    missing_att_count = df_clean["Attendance_Percent"].isnull().sum()
    df_clean["Attendance_Percent"] = df_clean["Attendance_Percent"].fillna(attendance_median)
    print(f"\nImputed {missing_att_count} missing Attendance_Percent values with median: {attendance_median:.2f}%")

    # Task 4: Check score columns for values outside [0, 100]
    score_cols = ["Python_Marks", "Maths_Marks", "Physics_Marks"]
    print("\nScore Range Validation (0 to 100):")
    for col in score_cols:
        invalid_scores = df_clean[(df_clean[col] < 0) | (df_clean[col] > 100)]
        print(f"  - {col}: {len(invalid_scores)} out-of-range records (Min={df_clean[col].min()}, Max={df_clean[col].max()})")

    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tasks 5 & 6: Calculate total and average marks, and assign Grade bands using pd.cut()."""
    print("\n" + "=" * 60)
    print("5–6. TOTAL, AVERAGE MARKS & GRADE DISTRIBUTION")
    print("=" * 60)

    score_cols = ["Python_Marks", "Maths_Marks", "Physics_Marks"]

    # Task 5: Calculate total and average marks
    df["Calculated_Total"] = df[score_cols].sum(axis=1)
    df["Average_Marks"] = (df["Calculated_Total"] / len(score_cols)).round(2)

    # Verify supplied Total_Marks against calculated sum
    mismatches = (df["Total_Marks"] != df["Calculated_Total"]).sum()
    print(f"Total_Marks verification mismatches: {mismatches}")

    # Task 6: Create Grade column using clearly defined score ranges
    grade_bins = [-np.inf, 40, 50, 60, 70, 80, 90, np.inf]
    grade_labels = [
        "F (<40%)",
        "D (40-50%)",
        "C (50-60%)",
        "B (60-70%)",
        "B+ (70-80%)",
        "A (80-90%)",
        "A+ (>=90%)",
    ]
    df["Grade"] = pd.cut(df["Average_Marks"], bins=grade_bins, labels=grade_labels, right=False)

    print("\nGrade Distribution:")
    print(df["Grade"].value_counts().sort_index())
    return df


def analyze_performance(df: pd.DataFrame) -> None:
    """Tasks 7–14: Department summary, top 10, at-risk students, attendance, study hours, internet, NumPy & IQR."""
    print("\n" + "=" * 60)
    print("7. DEPARTMENT-WISE AVERAGE MARKS")
    print("=" * 60)
    dept_summary = (
        df.groupby("Department")
        .agg(
            Student_Count=("Student_ID", "count"),
            Avg_Total=("Calculated_Total", "mean"),
            Avg_Percentage=("Average_Marks", "mean"),
            Median_Percentage=("Average_Marks", "median"),
            Avg_Python=("Python_Marks", "mean"),
            Avg_Maths=("Maths_Marks", "mean"),
            Avg_Physics=("Physics_Marks", "mean"),
            Avg_Attendance=("Attendance_Percent", "mean"),
            Avg_Study_Hours=("Study_Hours_Per_Day", "mean"),
        )
        .round(2)
        .sort_values(by="Avg_Percentage", ascending=False)
    )
    print(dept_summary)

    print("\n" + "=" * 60)
    print("8. TOP 10 STUDENTS BY AVERAGE MARKS")
    print("=" * 60)
    top_10 = df.sort_values(by=["Average_Marks", "Calculated_Total"], ascending=False).head(10)
    print(
        top_10[
            ["Student_ID", "Department", "Python_Marks", "Maths_Marks", "Physics_Marks", "Average_Marks", "Grade"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("9. STUDENTS BELOW 40% AVERAGE MARKS (AT-RISK CHECK)")
    print("=" * 60)
    below_40_avg = df[df["Average_Marks"] < 40]
    print(f"Students with Average_Marks < 40%: {len(below_40_avg)}")
    if not below_40_avg.empty:
        print(below_40_avg[["Student_ID", "Department", "Average_Marks", "Result"]])

    # Additional check: students failing individual subjects (<40) or marked Fail
    subject_fail = df[(df["Python_Marks"] < 40) | (df["Maths_Marks"] < 40) | (df["Physics_Marks"] < 40)]
    print(f"Students scoring < 40 in at least one subject: {len(subject_fail)}")
    print(f"Students with Result == 'Fail': {(df['Result'] == 'Fail').sum()}")

    print("\n" + "=" * 60)
    print("10. AVERAGE MARKS BY ATTENDANCE BANDS")
    print("=" * 60)
    att_bins = [0, 60, 75, 85, 100.1]
    att_labels = ["<60% (Low)", "60-75% (Moderate)", "75-85% (Good)", ">=85% (High)"]
    df["Attendance_Band"] = pd.cut(df["Attendance_Percent"], bins=att_bins, labels=att_labels, right=False)
    att_summary = (
        df.groupby("Attendance_Band", observed=False)
        .agg(
            Student_Count=("Student_ID", "count"),
            Avg_Marks=("Average_Marks", "mean"),
            Median_Marks=("Average_Marks", "median"),
            Avg_Study_Hours=("Study_Hours_Per_Day", "mean"),
        )
        .round(2)
    )
    print(att_summary)

    print("\n" + "=" * 60)
    print("11. STUDY HOURS VS AVERAGE MARKS")
    print("=" * 60)
    corr_study = df["Study_Hours_Per_Day"].corr(df["Average_Marks"])
    study_bins = [0, 2, 3.5, 5, np.inf]
    study_labels = ["<2 hrs", "2-3.5 hrs", "3.5-5 hrs", ">=5 hrs"]
    df["Study_Band"] = pd.cut(df["Study_Hours_Per_Day"], bins=study_bins, labels=study_labels, right=False)
    study_summary = (
        df.groupby("Study_Band", observed=False)
        .agg(Student_Count=("Student_ID", "count"), Avg_Marks=("Average_Marks", "mean"))
        .round(2)
    )
    print(f"Pearson Correlation (Study Hours vs Average Marks): {corr_study:.4f}")
    print(study_summary)

    print("\n" + "=" * 60)
    print("12. PERFORMANCE BY INTERNET ACCESS")
    print("=" * 60)
    inet_summary = (
        df.groupby("Internet_Access")
        .agg(
            Student_Count=("Student_ID", "count"),
            Avg_Marks=("Average_Marks", "mean"),
            Median_Marks=("Average_Marks", "median"),
            Avg_Python=("Python_Marks", "mean"),
            Avg_Study_Hours=("Study_Hours_Per_Day", "mean"),
        )
        .round(2)
    )
    print(inet_summary)

    print("\n" + "=" * 60)
    print("13. PYTHON MARKS STATISTICS (NUMPY)")
    print("=" * 60)
    python_scores = df["Python_Marks"].to_numpy()
    py_median = np.median(python_scores)
    py_std = np.std(python_scores, ddof=1)
    q1, q2, q3 = np.percentile(python_scores, [25, 50, 75])
    py_iqr = q3 - q1
    print(f"Median: {py_median:.2f}")
    print(f"Standard Deviation (sample): {py_std:.2f}")
    print(f"Q1 (25th percentile): {q1:.2f}")
    print(f"Q2 (50th percentile): {q2:.2f}")
    print(f"Q3 (75th percentile): {q3:.2f}")
    print(f"IQR (Q3 - Q1): {py_iqr:.2f}")

    print("\n" + "=" * 60)
    print("14. POTENTIAL SCORE OUTLIERS (IQR METHOD)")
    print("=" * 60)
    for col in ["Average_Marks", "Python_Marks", "Maths_Marks", "Physics_Marks"]:
        arr = df[col].to_numpy()
        col_q1, col_q3 = np.percentile(arr, [25, 75])
        col_iqr = col_q3 - col_q1
        lower_bound = col_q1 - 1.5 * col_iqr
        upper_bound = col_q3 + 1.5 * col_iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"{col} | Bounds: [{lower_bound:.2f}, {upper_bound:.2f}] | Outliers ({len(outliers)}): {list(outliers['Student_ID'])}")


def print_recommendations() -> None:
    """Task 15: Five data-supported recommendations for academic support."""
    print("\n" + "=" * 60)
    print("15. ACADEMIC SUPPORT RECOMMENDATIONS")
    print("=" * 60)
    recommendations = [
        "1. Target subject-level <40 scorers (31 students) and 'Fail' statuses (18 students) early, as overall average >=40% masks single-subject failures—especially in Maths (lowest mean: 67.21).",
        "2. Trigger proactive advisor check-ins for students below 60% attendance, whose average score (67.81%) trails peers in higher attendance bands.",
        "3. Focus mentoring on active study quality and assignment completion rather than raw study hours alone, given the near-zero linear correlation (r = 0.035) between study hours and marks.",
        "4. Provide offline learning resources and extended computer-lab access for the 58 students without home internet access.",
        "5. Conduct individual reviews for low IQR outliers (such as STU0144 and STU0395 on overall average) to diagnose specific conceptual or exam-attendance issues.",
    ]
    for rec in recommendations:
        print(rec)


def main() -> None:
    filepath = "02_student_performance.csv"
    df_raw = load_and_inspect(filepath)
    df_clean = check_quality_and_clean(df_raw)
    df_final = engineer_features(df_clean)
    analyze_performance(df_final)
    print_recommendations()

    # Export cleaned dataset for reproducibility
    df_final.to_csv("02_student_performance_cleaned.csv", index=False)
    print("\nSaved cleaned dataset to '02_student_performance_cleaned.csv'.")


if __name__ == "__main__":
    main()
