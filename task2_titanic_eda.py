import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


print("Loading Titanic Dataset...\n")

df = pd.read_csv("Titanic-Dataset.csv")



print("FIRST 5 ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nDATASET INFO:")
print(df.info())

print("\nSTATISTICAL SUMMARY:")
print(df.describe())


print("\nMISSING VALUES:")
print(df.isnull().sum())



df["Age"] = df["Age"].fillna(df["Age"].median())


df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


df["Cabin"] = df["Cabin"].fillna("Unknown")


df.drop_duplicates(inplace=True)

print("\nMISSING VALUES AFTER CLEANING:")
print(df.isnull().sum())


plt.figure(figsize=(6,4))

sns.countplot(
    x="Survived",
    data=df
)

plt.title("Survival Count")

plt.savefig("survival_count.png")

plt.show()



plt.figure(figsize=(6,4))

sns.countplot(
    x="Sex",
    data=df
)

plt.title("Gender Distribution")

plt.savefig("gender_distribution.png")

plt.show()



plt.figure(figsize=(6,4))

sns.countplot(
    x="Sex",
    hue="Survived",
    data=df
)

plt.title("Survival by Gender")

plt.savefig("survival_by_gender.png")

plt.show()



plt.figure(figsize=(6,4))

sns.countplot(
    x="Pclass",
    data=df
)

plt.title("Passenger Class Distribution")

plt.savefig("class_distribution.png")

plt.show()



plt.figure(figsize=(6,4))

sns.countplot(
    x="Pclass",
    hue="Survived",
    data=df
)

plt.title("Survival by Passenger Class")

plt.savefig("survival_by_class.png")

plt.show()


plt.figure(figsize=(8,5))

sns.histplot(
    df["Age"],
    bins=20,
    kde=True
)

plt.title("Age Distribution")

plt.savefig("age_distribution.png")

plt.show()


plt.figure(figsize=(8,5))

sns.histplot(
    df["Fare"],
    bins=20,
    kde=True
)

plt.title("Fare Distribution")

plt.savefig("fare_distribution.png")

plt.show()


numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.show()



print("\nINSIGHTS")

female_survival = (
    df[df["Sex"] == "female"]["Survived"].mean()
    * 100
)

male_survival = (
    df[df["Sex"] == "male"]["Survived"].mean()
    * 100
)

print(f"\nFemale Survival Rate: {female_survival:.2f}%")
print(f"Male Survival Rate: {male_survival:.2f}%")

print("\nKey Findings:")
print("1. Female passengers had a higher survival rate.")
print("2. First-class passengers survived more frequently.")
print("3. Most passengers were between 20 and 40 years old.")
print("4. Higher ticket fares were associated with better survival chances.")
print("5. Passenger class had a strong influence on survival.")

