"""Prepare the bike-sharing tutorial dataset.

Turns the raw UCI Bike Sharing dataset (hour.csv) into a single clean,
human-readable CSV (bikes.csv) used by the week-4 tutorial.

Raw data: Fanaee-T, H. & Gama, J. (2014). Event labeling combining ensemble
detectors and background knowledge. Progress in Artificial Intelligence.
Downloaded from the UCI Machine Learning Repository (CC BY 4.0):
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Transformations (documented so curious students can follow along):
- De-normalize weather variables back to physical units
  (UCI ships temp/41, atemp/50, hum/100, windspeed/67).
- Replace integer codes by readable labels (season, weather, weekday, month).
  Note: UCI labels season 1 as "spring" but it covers Jan-Mar; we relabel by
  actual calendar season (1=winter, 2=spring, 3=summer, 4=fall).
- Drop columns that would leak the target (casual + registered = rentals)
  and the row index (instant).
"""

import pandas as pd

SEASONS = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'}
WEATHER = {1: 'clear', 2: 'mist', 3: 'light_rain_snow', 4: 'heavy_rain_snow'}
WEEKDAYS = {0: 'sunday', 1: 'monday', 2: 'tuesday', 3: 'wednesday',
            4: 'thursday', 5: 'friday', 6: 'saturday'}
MONTHS = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
          7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'}


def main(raw_path='hour.csv', out_path='bikes.csv'):
    df = pd.read_csv(raw_path)

    out = pd.DataFrame({
        'date': df['dteday'],
        'season': df['season'].map(SEASONS),
        'year': df['yr'] + 2011,
        'month': df['mnth'].map(MONTHS),
        'hour': df['hr'],
        'weekday': df['weekday'].map(WEEKDAYS),
        'workingday': df['workingday'],
        'holiday': df['holiday'],
        'weather': df['weathersit'].map(WEATHER),
        'temp': (df['temp'] * 41).round(1),          # degrees Celsius
        'feels_like': (df['atemp'] * 50).round(1),   # degrees Celsius
        'humidity': (df['hum'] * 100).round(0).astype(int),   # percent
        'windspeed': (df['windspeed'] * 67).round(1),         # km/h
        'rentals': df['cnt'],
    })

    out.to_csv(out_path, index=False)
    print(f'Wrote {out_path}: {out.shape[0]} rows, {out.shape[1]} columns')
    print(out.head())


if __name__ == '__main__':
    main()
