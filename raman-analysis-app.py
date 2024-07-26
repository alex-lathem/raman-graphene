### Imports
import streamlit as st
import rampy as rp
from BaselineRemoval import BaselineRemoval
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='talk', style='ticks', palette='colorblind')

TURQUOISE = "#0eaf9d"
TRANSPARENT_TURQUOISE = "#0eaf9d88"
PURPLE = "#9d0eaf"

### Functions
# Baseline Correction
def back_subt(point):
    x = point['Wave Number']
    y = point['Intensity']
    # Create array to store corrected y
    y = rp.smooth(x,y,method="savgol", window_length=5, polynomial_order=4)
    # try:
    #     baseObj = BaselineRemoval(y)
    #     y_corr = baseObj.ZhangFit()
    # except:
    #     baseObj = BaselineRemoval(y)
    #     y_corr = baseObj.ModPoly(2)
    
    baseObj = BaselineRemoval(y)
    y_corr = baseObj.ZhangFit()
    
    point['Corrected Intensity'] = y_corr

    return point

@st.cache_data
def unpack_dataframe(file_buffer, correct_spectrum=True):
    df = pd.read_csv(file_buffer, sep='\s+', skipinitialspace=True, names=['X', 'Y', 'Wave Number', 'Intensity'], skiprows=[0])
    if df.isna().any().any():
        return None
    
    if correct_spectrum:
        df_grouped = df.groupby(['X', 'Y']).apply(back_subt)
    else:
        df_grouped = df.groupby(['X', 'Y']).apply(lambda x: x)
        df_grouped['Corrected Intensity'] = df_grouped['Intensity']
    df_grouped.reset_index(drop=True, inplace=True)

    return df_grouped

def get_mean_spectrum(df):
    if df is None:
        return None
    
    mean_spectrum = df.groupby('Wave Number')['Corrected Intensity'].agg(['mean', 'std']).reset_index()
    mean_spectrum.columns = ['Wave Number', 'Mean Intensity', 'Std']
    
    return mean_spectrum

def plot_mean_spectrum(df, color=TURQUOISE, title_string=''):
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='Wave Number', y='Mean Intensity', color=color, ax=ax).set(ylabel='Intensity', title=title_string)
    plt.fill_between(
        df['Wave Number'], np.subtract(df['Mean Intensity'],
        df['Std']), np.add(df['Mean Intensity'], df['Std']),
        alpha=0.2, linewidth=4, linestyle='solid', antialiased=True, color=color)

    plt.yticks([])

    st.pyplot(fig)

def replace_extension(str, new_extension):
    return str[:-4] + new_extension

def show_mean_spectrum(file, correct_spectrum=True):
    mean_spectrum = get_mean_spectrum(unpack_dataframe(file, correct_spectrum=correct_spectrum))
    if mean_spectrum is not None:
        st.table(mean_spectrum)
        plot_mean_spectrum(mean_spectrum, title_string=file.name)

def D_peak_max(point):
    return point[(point['Wave Number']>1250) & (point['Wave Number']<1450)]['Corrected Intensity'].max()

def G_peak_max(point):
    return point[(point['Wave Number']>1500) & (point['Wave Number']<1700)]['Corrected Intensity'].max()

def pos_2D_max(point):
    return point['Wave Number'].iloc[point[(point['Wave Number']>2500) & (point['Wave Number']<2801)]['Corrected Intensity'].argmax()]

def peak_2D_max(point):
    return point[(point['Wave Number']>2500) & (point['Wave Number']<2801)]['Corrected Intensity'].max()

def fwhm_2D(point):
    try:
        spectrum_window = point[(point['Wave Number']>2500) & (point['Wave Number']<2801)]['Corrected Intensity']
        peaks, properties = find_peaks(spectrum_window, height=10, distance=20, width=15)
        widths, width_heights, left_ips, right_ips = peak_widths(spectrum_window, peaks, rel_height=0.5)

        return widths[0]
        
    except:
        return np.nan

def point_noise(point):
    return point[(point['Wave Number']>2100) & (point['Wave Number']<2300)]['Corrected Intensity'].std()

def determine_if_carbon(point, noiseThreshold_G):
    return point['SNR G'] > noiseThreshold_G

def determine_if_graphene(point):
    return point['Is Carbon'] and point['Max 2D/G'] > 0.3 and point['SNR 2D'] > 5 and point['Max 2D/G'] < 20 and not np.isnan(point['2D FWHM'])

def process_map(df, filename, noiseThreshold_G):

    # Group by point to get individual spectra
    df_grouped = df.groupby(['X', 'Y'])

    # Calculate quantities which need the spectrum information of a given point
    D_max = df_grouped.apply(D_peak_max)
    G_max = df_grouped.apply(G_peak_max)
    TwoD_max = df_grouped.apply(peak_2D_max)
    pos_2D = df_grouped.apply(pos_2D_max)
    noise = df_grouped.apply(point_noise)
    TwoD_FWHM = df_grouped.apply(fwhm_2D)

    # Reform the grouped table into a reindexed DataFrame
    point_stats = pd.DataFrame([D_max, G_max, TwoD_max, TwoD_FWHM, pos_2D, noise]).T.reset_index()
    point_stats.columns = ['X', 'Y', 'Max D', 'Max G', 'Max 2D', '2D FWHM', 'Pos 2D', 'Noise']

    # Calculate statistics which do not need full point spectra
    point_stats['Max D/G'] = point_stats['Max D']/point_stats['Max G']
    point_stats['Max 2D/G'] = point_stats['Max 2D']/point_stats['Max G']
    point_stats['SNR G'] = point_stats['Max G']/point_stats['Noise']
    point_stats['SNR 2D'] = point_stats['Max 2D']/point_stats['Noise']

    # Determine whether each point is considered graphene and/or carbon of some kind
    point_stats['Is Carbon'] = point_stats.apply(lambda point: determine_if_carbon(point, noiseThreshold_G), axis=1)
    point_stats['Is Graphene'] = point_stats.apply(determine_if_graphene, axis=1)

    # Calculate graphene yield and average stats for whole file
    graphene_yield = pd.Series(point_stats['Is Graphene'].sum()/point_stats['Is Carbon'].sum(), index=['Graphene Yield'])
    file_stats = point_stats[['Max 2D/G', 'Max D/G', '2D FWHM', 'Pos 2D', 'SNR 2D', 'SNR G']].mean()
    file_stats = pd.concat([pd.Series(filename, index=['Filename']), graphene_yield, file_stats])

    return point_stats, file_stats

### App start
st.title("Raman Analysis")

uploaded_files = st.file_uploader(
    "Upload Raman map files", accept_multiple_files=True
)

noiseThreshold_G = st.slider("Signal-to-noise threshold for G peak", 2, 30, 8)

for f in uploaded_files:
    st.subheader(f"Stats for {f.name}")

    data_table = unpack_dataframe(f)
    point_stats, file_stats = process_map(data_table, f.name, noiseThreshold_G)
    st.write("Mean spectrum for map:")
    mean_spec = get_mean_spectrum(data_table)
    plot_mean_spectrum(mean_spec)

    plot_csv = mean_spec.to_csv().encode("utf-8")

    st.download_button(
        label=":arrow_down: Download histogram data as CSV",
        data=plot_csv,
        file_name=f"MEAN_SPECTRUM_{f.name[:-4]}.csv",
        mime="text/csv",
        key=f"download_mean_{f.name}"
    )

    if st.checkbox("Show point statistics", key=f"ps_{f.name}"):
        st.write(point_stats)
    if st.checkbox("Show file stats", key=f"fs_{f.name}"):
        st.write(file_stats)

    st.write(f'{f.name} processed.')
