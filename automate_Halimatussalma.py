
import pandas as pd

def preprocess_titanic(input_path='train.csv', output_path='processed_train.csv'):
    df = pd.read_csv(input_path)
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    df.to_csv(output_path, index=False)
    print(f"Hasil preprocessing disimpan ke: {output_path}")

if __name__ == '__main__':
    preprocess_titanic()
