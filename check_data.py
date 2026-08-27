import pandas as pd

file_path = "data/student_data.csv"

df = pd.read_csv(file_path)

print("DATASET LOADED SUCCESSFULLY")
print()
print("Number of rows:", len(df))
print("Number of columns:", len(df))
print()
print("Column names:")
print(df.columns.tolist())
print()
print("First 5 records:")
print(df.head())
import pandas as pd

file_path = "data/student_data.csv"

df = pd.read_csv(file_path)

print("===== DATASET QUALITY CHECK =====")
print()

print("Rows:", len(df))
print("Columns:", len(df.columns))
print()

print("Missing values:")
print(df.isnull().sum())
print()

print("Duplicate rows:", df.duplicated().sum())
print()

print("Data types:")
print(df.dtypes)
import pandas as pd

file_path = "data/student_data.csv"

df = pd.read_csv(file_path)

print("===== DATA RANGE CHECK =====")
print()

print("Minimum values:")
print(df.min())
print()

print("Maximum values:")
print(df.max())
print()

print("Average values:")
print(df.mean())
import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/student_data.csv"

df = pd.read_csv(file_path)

print("===== DATASET VISUALIZATION =====")
print()

print("Dataset loaded successfully.")
print("Total students:", len(df))

# Attendance vs Final Marks
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Attendance"],
    df["FinalMarks"]
)

plt.xlabel("Attendance (%)")
plt.ylabel("Final Marks")
plt.title("Attendance vs Final Marks")

plt.grid(True)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/student_data.csv"

df = pd.read_csv(file_path)

print("===== STUDY HOURS ANALYSIS =====")
print()
print("Dataset loaded successfully.")
print("Total students:", len(df))

# Study Hours vs Final Marks
plt.figure(figsize=(8, 5))

plt.scatter(
    df["StudyHours"],
    df["FinalMarks"]
)

plt.xlabel("Study Hours per Day")
plt.ylabel("Final Marks")
plt.title("Study Hours vs Final Marks")

plt.grid(True)
plt.show()