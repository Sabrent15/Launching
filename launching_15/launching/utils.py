from datetime import datetime, timedelta
import numpy as np

def getidx():
    return {
        'a': 0,
        'ecco': 1,
        'inclo': 2,
        'nodeo': 3,
        'argpo': 4,
        'mo': 5,
        'bstar': 6,
        'mass': 7,
        'radius': 8,
        'error': 9,
        'controlled': 10,
        'a_desired': 11,
        'missionlife': 12,
        'constel': 13,
        'date_created': 14,
        'launch_date': 15,
        'r': [16, 17, 18],
        'v': [19, 20, 21],
        'objectclass': 22,
        'ID': 23
    }

def jd2date(jd_array):
    """
    Converts Julian Dates to datetime objects.
    Accepts a scalar or numpy array.
    """
    jd_array = np.atleast_1d(jd_array)
    date_array = []

    for jd in jd_array:
        jd = float(jd)
        jd += 0.5
        Z = int(jd)
        F = jd - Z
        if Z < 2299161:
            A = Z
        else:
            alpha = int((Z - 1867216.25) / 36524.25)
            A = Z + 1 + alpha - int(alpha / 4)
        B = A + 1524
        C = int((B - 122.1) / 365.25)
        D = int(365.25 * C)
        E = int((B - D) / 30.6001)

        day = B - D - int(30.6001 * E) + F
        if E < 14:
            month = E - 1
        else:
            month = E - 13
        if month > 2:
            year = C - 4716
        else:
            year = C - 4715

        day_frac = day - int(day)
        hour = int(day_frac * 24)
        minute = int((day_frac * 24 - hour) * 60)
        second = int((((day_frac * 24 - hour) * 60) - minute) * 60)
        date_array.append(datetime(year, month, int(day), hour, minute, second))

    return np.array(date_array) if len(date_array) > 1 else date_array[0]
