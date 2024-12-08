

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler
import unidecode
# импорты

df = pd.read_excel('football_dataset.xlsx', engine='openpyxl')
df.head()

def clean_soccer_data(df):
    # копия, чтобы ориг не трогать
    cleaned_df = df.copy()

    # убираем тех, у кого возраст = 0
    cleaned_df = cleaned_df[cleaned_df['age'] != 0]

    # функция, чтобы заменить все не англ элементы
    def replace_non_english_chars(text):
        try:
            if isinstance(text, str):
                # unidecode, чтобы взять ближайшее по звуку
                return unidecode.unidecode(text)
            return text
        except:
            return text

    # замена по командам и игрокам
    if 'squad' in cleaned_df.columns:
        cleaned_df['squad'] = cleaned_df['squad'].apply(replace_non_english_chars)
    if 'player' in cleaned_df.columns:
        cleaned_df['player'] = cleaned_df['player'].apply(replace_non_english_chars)

    return cleaned_df

try:
    # прочесть
    df = pd.read_excel('football_dataset.xlsx')

    # очистить
    cleaned_df = clean_soccer_data(df)

    # сохранить
    cleaned_df.to_csv('cleaned_soccer_players.csv', index=False)

    print("Dataset cleaned")

except Exception as e:
    print(f"error: {str(e)}")

df = cleaned_df
df

# спред по лигам
df.groupby('league').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

# стоимости
df['value'].plot(kind='hist', bins=20, title='value')
plt.gca().spines[['top', 'right',]].set_visible(False)

# диаграмма рассеяния с линией тренда
# размер графика
plt.figure(figsize=(10,6))

# разбросанные точки по позициям
sns.scatterplot(
    data=df,
    x='minutes',
    y='value',
    hue='position',
    alpha=0.6 # прозрачность точек
)

# линия тренда
sns.regplot(
    data=df,
    x='minutes',
    y='value',
    scatter=False, # без точек, только линия
    color='black'
)

# подписи
plt.title('Minutes Played / Market Vale by Position')
plt.xlabel('Mins Played')
plt.ylabel('Market Value')

# показать график
st.pyplot(plt)

# размер
plt.figure(figsize=(10,6))

# коэфф попадания в старт
df['start_coef'] = df['games_starts'] / df['games']

# базовая статистика
stats = df.agg({
    'start_coef': ['mean', 'median', 'std'],
    'games': ['mean', 'median', 'std'],
    'value': ['mean', 'median', 'std']
})

# группировка игроков по времени в старте
df['starter_type'] = pd.qcut(df['start_coef'],
                           q=4,
                           labels=['Rare Starter',
                                   'Oсasional Starter',
                                   'Regular Starter',
                                   'Key Starter'])

# строим график
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='starter_type', y='value')

# оформление
plt.xticks(rotation=45)
plt.title('Player Value by Role in Team')
plt.xlabel('Player Role')
plt.ylabel('Market Value')

st.pyplot(plt)

# счет позиций: кол запятых + 1
df['position_count'] = df['position'].str.count(',') + 1

# диаграмма скрипка: в центре медиана, тело свечи - диапозон, фетиль - интервал, ширина - частота
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='position_count', y='value', color='c')
plt.title('Market Value Distrebution by Number of Positions')
plt.xlabel('Number of Positions')
plt.ylabel('Market Value')

# средние знач сравним
plt.figure(figsize=(10, 6))
avg_values = df.groupby('position_count')['value'].mean()
avg_values.plot(kind='bar')
plt.title('Average Market Value by Number of Positions')
plt.xlabel('Number of Positions')
plt.ylabel('Average Market Value')

age_counts = df['age'].value_counts()

# bar plot
age_counts.plot(kind='bar', color='skyblue')

# заголовок и метки осей
plt.title('Age distribution')
plt.xlabel('Age')

plt.ylabel('Am of players')

# показать
plt.show()

# bar plot ср стоимости по возрасту
age_groups = df.groupby('age')['value'].mean().reset_index()
sns.barplot(data=age_groups, x='age', y='value')
plt.title('Average Market Value by Age')
plt.xlabel('Age')
plt.ylabel('Average Market Value')

plt.tick_params(axis='x', rotation=45)
plt.grid(True, alpha=0.3)

# показать
st.pyplot(plt)

# категории по игровому времени
df['playing_time_ratio'] = df['minutes'] / df['games']
df['playing_time_category'] = pd.qcut(df['playing_time_ratio'],
                                    q=3,
                                    labels=['Low Minutes', 'Medium Minutes', 'High Minutes'])

# базовая статистика по числовым полям
numerical_stats = df.agg({
    'minutes': ['mean', 'median', 'std'],
    'value': ['mean', 'median', 'std'],
    'games': ['mean', 'median', 'std']
}).round(2)

print("Descriptive Statistics:")
print(numerical_stats)

# делаем большой график с 4 подграфиками
plt.figure(figsize=(15, 15))


# график 1 - диаграмма рассеяния возраст/стоимость
plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='age', y='value', alpha=0.5)
sns.regplot(data=df, x='age', y='value', scatter=False, color='red')

plt.title('Age / Market Value')

plt.xlabel('Age')
plt.ylabel('Market Value')

# график 2 - ящики с усами по возрастным группам
df['age_group'] = pd.cut(df['age'],
                        bins=[16, 23, 27, 30, 33, 100],
                        labels=['Under 23', '23-27', '27-30', '30-33', 'Over 33'])

plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='age_group', y='value', hue='playing_time_category')

plt.title('Value Distribution by Age Group and Playing Time')
plt.xticks(rotation=45)
plt.ylabel('Market Value')

# график 3 - тепловая карта средних значений
plt.subplot(2, 2, 3)
pivot_table = df.pivot_table(values='value',
                           index='age_group',
                           columns='playing_time_category',
                           aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('Avg Values by Age and Playing Time')

# график 4 - линейный график трендов
avg_values = df.groupby(['age_group', 'playing_time_category'])['value'].mean().unstack()
avg_values.plot(marker='o')
plt.title('Val Trends by Age Group and Playing Time')

plt.xlabel('Age Group')

plt.ylabel('Average Value')
plt.xticks(rotation=45)

plt.tight_layout()
st.pyplot(plt)


# детальная стата по группам
detailed_stats = df.groupby(['age_group', 'playing_time_category']).agg({
    'value': ['mean', 'median', 'std'],
    'minutes': ['mean', 'median', 'std'],
    'games': ['mean', 'median', 'std']
}).round(2)

print("\nDetailed Stats by Age Gr and Playing Time:")
print(detailed_stats)

# считаем долю стартовых матчей
df['start_ratio'] = df['games_starts'] / df['games']

# добавляем коэфф для специалистов
df['value'] = df['value'] * (1 + df['start_ratio'] * 0.2)

# разделяем на специалистов и ротац игроков
df['player_type'] = df['start_ratio'].apply(
    lambda x: 'Specialist' if x >= 0.75 else 'Rotator'
)

# смотрим корреляции
print("\nCorelation Analysis")
correlations = df[['value', 'start_ratio', 'minutes']].corr()
print("\nCorrelations:")
print(correlations)

# разбиваем по ценовым категориям
df['value_bracket'] = pd.qcut(df['value'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# считаем процент типов игроков в каждой ценовой категории
print("\nValue Bracket Analysis", )
value_bracket_analysis = pd.crosstab(df['value_bracket'], df['player_type'], normalize='index') * 100
print("\nPercentage of player types in each value bracket:")
print(value_bracket_analysis)

# добавляем нужные столбцы
def prepare_data(df):
    # считаем кол поз
    df['versatility'] = df['position'].str.count(',') + 1

    # доля матчей в старте
    df['starter_ratio'] = df['games_starts'] / df['games']
    df['starter_category'] = pd.cut(df['starter_ratio'],
                                  [0, 0.25, 0.5, 0.75, 1],
                                  labels=['Rare', 'Occasional', 'Regular', 'Key'])

    # категории по количеству позиций
    df['versatility_category'] = df['versatility'].map({
        1: 'Single-position',
        2: 'Dual-position',
    })
    return df

