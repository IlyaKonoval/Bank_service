import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        clients = pd.read_csv(os.path.join(self.data_path, "D_clients.csv"))
        job = pd.read_csv(os.path.join(self.data_path, "D_job.csv"))
        salary = pd.read_csv(os.path.join(self.data_path, "D_salary.csv"))
        target = pd.read_csv(os.path.join(self.data_path, "D_target.csv"))
        cl_loan = pd.read_csv(os.path.join(self.data_path, "D_close_loan.csv"))
        loan = pd.read_csv(os.path.join(self.data_path, "D_loan.csv"))
        return clients, job, salary, target, cl_loan, loan

    def merge_data(self, clients, job, salary, target, cl_loan, loan):
        dataset_client_target = pd.merge(clients, target, left_on='ID', right_on='ID_CLIENT')
        dataset_client_target_salary = pd.merge(dataset_client_target, salary, left_on='ID', right_on='ID_CLIENT')
        dataset_client_target_salary_job = pd.merge(dataset_client_target_salary, job, left_on='ID', right_on='ID_CLIENT')
        dataset_client_target_salary_job.drop('ID_CLIENT_x', axis=1, inplace=True)

        closed_loans = pd.merge(cl_loan, loan, on='ID_LOAN', how='left')
        closed_loans = closed_loans[closed_loans['CLOSED_FL'] == 1]
        loan_closed = closed_loans.groupby('ID_CLIENT').size().reset_index(name='LOAN_NUM_CLOSED')
        loan_total = loan.groupby('ID_CLIENT').size().reset_index(name='LOAN_NUM_TOTAL')
        loan_data = pd.merge(loan_total, loan_closed, on='ID_CLIENT', how='left')

        dataset_client_target_salary_job_loan = pd.merge(dataset_client_target_salary_job, loan_data, left_on='ID', right_on='ID_CLIENT')

        return dataset_client_target_salary_job_loan

    def clean_data(self, df):
        columns_to_drop = ['EDUCATION', 'MARITAL_STATUS', 'REG_ADDRESS_PROVINCE', 'ID_CLIENT_x', 'ID_CLIENT_y',
                           'JOB_DIR', 'WORK_TIME', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE', 'FAMILY_INCOME',
                           'OWN_AUTO', 'FL_PRESENCE_FL', 'ID', 'GEN_TITLE', 'GEN_INDUSTRY']
        df = df.drop(columns_to_drop, axis=1)
        df['LOAN_NUM_CLOSED'] = df['LOAN_NUM_CLOSED'].fillna(0)
        new_order = ['AGE', 'GENDER', 'CHILD_TOTAL', 'DEPENDANTS', 'PERSONAL_INCOME', 'SOCSTATUS_WORK_FL',
                     'SOCSTATUS_PENS_FL', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED', 'TARGET']
        df = df[new_order]
        df = df.drop_duplicates()
        df = df.loc[df.PERSONAL_INCOME >= 5000]
        df = df.loc[df.LOAN_NUM_TOTAL >= df.LOAN_NUM_CLOSED]
        return df

    def process_data(self):
        clients, job, salary, target, cl_loan, loan = self.load_data()
        self.df = self.merge_data(clients, job, salary, target, cl_loan, loan)
        self.df = self.clean_data(self.df)
        return self.df

    def save_data(self, output_path):
        """Сохраняет обработанный датасет в CSV файл."""
        parent_dir = os.path.dirname(output_path)
        output_file = os.path.join(parent_dir, "processed_data.csv")
        self.df.to_csv(output_file, index=False)



class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def analyze_data(self):
        self.numerical_analysis()
        self.categorical_analysis()
        self.correlation_analysis()
        self.target_analysis()

    def numerical_analysis(self):
        """Анализ числовых признаков."""

        # Возраст (AGE)
        plt.figure(figsize=(7, 5))
        plt.hist(self.df.AGE, bins=30)
        plt.title('Распределение возраста клиентов')
        plt.xlabel('Возраст')
        plt.ylabel('Количество клиентов')
        plt.show()

        print(f"Доля клиентов старше 30 лет: {len(self.df[self.df.AGE > 30]) / len(self.df):.2f}")
        print(f"Средний возраст клиентов: {self.df.AGE.mean():.2f}")
        print(f"Максимальный возраст: {self.df.AGE.max()}")
        print(f"Минимальный возраст: {self.df.AGE.min()}")

        plt.figure(figsize=(6, 4))
        plt.scatter(self.df.AGE, self.df.TARGET)
        plt.title('Зависимость отклика от возраста')
        plt.xlabel('Возраст')
        plt.ylabel('Отклик (TARGET)')
        plt.show()

        print(f"Корреляция возраста и отклика (Пирсон): {np.corrcoef(self.df.AGE, self.df.TARGET)[0][1]:.2f}")

        # Персональный доход (PERSONAL_INCOME)
        plt.figure(figsize=(10, 7))
        plt.hist(self.df.PERSONAL_INCOME, bins=100)
        plt.title('Распределение персонального дохода')
        plt.xlabel('Персональный доход')
        plt.ylabel('Количество клиентов')
        plt.show()

        print(f"Средний персональный доход: {self.df.PERSONAL_INCOME.mean():.2f}")
        print(f"Минимальный персональный доход: {self.df.PERSONAL_INCOME.min()}")
        print(f"Максимальный персональный доход: {self.df.PERSONAL_INCOME.max()}")


        # Количество детей (CHILD_TOTAL)
        plt.figure(figsize=(7, 5))
        plt.hist(self.df.CHILD_TOTAL, bins=20)
        plt.title('Распределение количества детей')
        plt.xlabel('Количество детей')
        plt.ylabel('Количество клиентов')
        plt.show()

        print(f"Доля клиентов без детей: {len(self.df[self.df.CHILD_TOTAL == 0]) / len(self.df):.2f}")

        plt.figure(figsize=(6, 4))
        plt.scatter(self.df.CHILD_TOTAL, self.df.DEPENDANTS)
        plt.title('Зависимость количества иждивенцев от количества детей')
        plt.xlabel('Количество детей')
        plt.ylabel('Количество иждивенцев')
        plt.show()

        print(f"Корреляция количества детей и количества иждивенцев (Пирсон): {np.corrcoef(self.df.CHILD_TOTAL, self.df.DEPENDANTS)[0][1]:.2f}")


        # Количество иждивенцев (DEPENDANTS)
        plt.figure(figsize=(7, 5))
        plt.hist(self.df.DEPENDANTS, bins=10)
        plt.title('Распределение количества иждивенцев')
        plt.xlabel('Количество иждивенцев')
        plt.ylabel('Количество клиентов')
        plt.show()

        # Общее количество ссуд (LOAN_NUM_TOTAL)
        plt.figure(figsize=(7, 5))
        plt.hist(self.df.LOAN_NUM_TOTAL, bins=20)
        plt.title('Распределение общего количества ссуд')
        plt.xlabel('Общее количество ссуд')
        plt.ylabel('Количество клиентов')
        plt.show()

        print(f"Доля клиентов с 1 кредитом: {len(self.df[self.df.LOAN_NUM_TOTAL == 1]) / len(self.df):.2f}")
        print(f"Максимальное количество ссуд у одного клиента: {self.df.LOAN_NUM_TOTAL.max()}")

        # Количество погашенных ссуд (LOAN_NUM_CLOSED)
        plt.figure(figsize=(7, 5))
        plt.hist(self.df.LOAN_NUM_CLOSED, bins=20)
        plt.title('Распределение количества погашенных ссуд')
        plt.xlabel('Количество погашенных ссуд')
        plt.ylabel('Количество клиентов')
        plt.show()

    def categorical_analysis(self):
        """Анализ категориальных признаков."""

        # Пол клиента (GENDER)
        gender_clients = self.df['GENDER'].value_counts()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=gender_clients.index, y=gender_clients.values, hue=gender_clients.index, palette='summer')
        plt.title('Разделение клиентов по полу')
        plt.xlabel('Пол')
        plt.ylabel('Количество')
        plt.show()

        print(gender_clients)

        print("Средний доход по полу:")
        print(self.df.groupby('GENDER').agg(IncomeMean=('PERSONAL_INCOME', 'mean')).sort_values(by='IncomeMean', ascending=False))

        plt.figure(figsize=(7, 5))
        sns.barplot(x='GENDER', y='TARGET', data=self.df, estimator=lambda x: sum(x) / len(x),hue='GENDER', palette='summer')
        plt.title('Вероятность отклика (TARGET) в зависимости от пола (GENDER)')
        plt.xlabel('Пол')
        plt.ylabel('Вероятность отклика (TARGET = 1)')
        plt.show()

        # Социальный статус клиента относительно работы (SOCSTATUS_WORK_FL)
        work_clients = self.df['SOCSTATUS_WORK_FL'].value_counts()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=work_clients.index, y=work_clients.values, hue=work_clients.index, palette='summer')
        plt.title('Разделение клиентов по наличию работы')
        plt.xlabel('Работает ли')
        plt.ylabel('Количество')
        plt.show()

        print(work_clients)

        print("Средний доход в зависимости от рабочего статуса:")
        print(self.df.groupby('SOCSTATUS_WORK_FL').agg(IncomeMean=('PERSONAL_INCOME', 'mean')).sort_values(by='IncomeMean', ascending=False))

        plt.figure(figsize=(7, 5))
        sns.barplot(x='SOCSTATUS_WORK_FL', y='TARGET', data=self.df, estimator=lambda x: sum(x) / len(x), hue='SOCSTATUS_WORK_FL' , palette='summer')
        plt.title('Вероятность отклика (TARGET) в зависимости от рабочего статуса (SOCSTATUS_WORK_FL)')
        plt.xlabel('Работает ли')
        plt.ylabel('Вероятность отклика (TARGET = 1)')
        plt.show()

        # Социальный статус клиента относительно пенсии (SOCSTATUS_PENS_FL)
        pens_clients = self.df['SOCSTATUS_PENS_FL'].value_counts()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=pens_clients.index, y=pens_clients.values, hue=pens_clients.index, palette='summer')
        plt.title('Разделение клиентов по наличию пенсии')
        plt.xlabel('На пенсии ли')
        plt.ylabel('Количество')
        plt.show()

        print(pens_clients)

        pens_by_gender = pd.crosstab(self.df['GENDER'], self.df['SOCSTATUS_PENS_FL'])
        pens_by_gender.plot(kind='bar', stacked=True)
        plt.title('Количество пенсионеров по полу')
        plt.xlabel('Пол')
        plt.ylabel('Количество')
        plt.legend(title='Статус пенсионера')
        plt.show()

        plt.figure(figsize=(7, 5))
        sns.barplot(x='SOCSTATUS_PENS_FL', y='TARGET', data=self.df, estimator=lambda x: sum(x) / len(x), hue='SOCSTATUS_PENS_FL', palette='summer', legend=False)
        plt.title('Вероятность отклика (TARGET) в зависимости от пенсионного статуса (SOCSTATUS_PENS_FL)')
        plt.xlabel('На пенсии ли')
        plt.ylabel('Вероятность отклика (TARGET = 1)')
        plt.show()


    def correlation_analysis(self):
        """Анализ корреляции признаков."""

        # Корреляция Пирсона
        corr_mat = self.df.corr()
        sns.heatmap(corr_mat, cmap='BuPu', annot=True)
        plt.title('Матрица корреляции (Пирсон)')
        plt.show()

        print("Матрица корреляции (Пирсон):")
        print(corr_mat)

        # Корреляция Спирмена
        corr_mat_spearman = self.df.corr(method='spearman')
        sns.heatmap(corr_mat_spearman, cmap='BuPu', annot=True)
        plt.title('Матрица корреляции (Спирмен)')
        plt.show()

        print("\nМатрица корреляции (Спирмен):")
        print(corr_mat_spearman)


    def target_analysis(self):
        """Анализ целевой переменной."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['TARGET'])
        plt.title('Распределение целевой переменной (TARGET)')
        plt.xlabel('Отклик (TARGET)')
        plt.ylabel('Количество клиентов')
        plt.show()

        print(f"Доля клиентов, давших отклик: {len(self.df[self.df.TARGET == 1]) / len(self.df):.2f}")

def get_universal_data_path():
    """Возвращает универсальный путь к папке 'data'. """

    # Проверяем, установлена ли переменная окружения DATA_PATH
    env_data_path = os.environ.get("DATA_PATH")
    if env_data_path:
        return env_data_path

    # Если переменная окружения не установлена, используем относительный путь
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def main():
    data_path = get_universal_data_path()
    processor = DataProcessor(data_path)
    df = processor.process_data()

    processor.save_data(data_path)

    analyzer = DataAnalyzer(df)
    analyzer.analyze_data()


if __name__ == "__main__":
    main()