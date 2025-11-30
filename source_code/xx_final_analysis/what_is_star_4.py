from uncertainties import ufloat
from uncertainties.umath import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

### This file performs final analysis for Star 4 and compares it with other variable star types - on HR and Luminosity-Period diagrams, and overlays their spectra

class VariableStarType:
    def __init__(self, name, type, period, temperature, bp_rp, j_k, j_h, w1_w2, w2_w3, w3_w4):
        self.name = name
        self.type = type
        self.period = period
        self.temperature = temperature
        self.bp_rp = bp_rp
        self.j_k = j_k
        self.j_h = j_h
        self.w1_w2 = w1_w2
        self.w2_w3 = w2_w3
        self.w3_w4 = w3_w4


def star_4():
    print("Analysis for Star 4:")

    period = ufloat(345.4, 16.4)  # days
    print(f'Period: {period} days')

    temp_eff = ufloat(5041, 20)  # Kelvin
    print(f'Effective Temperature: {temp_eff} K')

    print('----------------------ASAS-SN------------------------')

    app_mag_asassn_mean = ufloat(15.43, 0.233)  # apparent magnitude
    print(f'Average Apparent Magnitude ASAS-SN: {app_mag_asassn_mean}')

    app_mag_asassn_amplitude = ufloat(1.03, 0.466)  # amplitude
    print(f'Amplitude Apparent Magnitude ASAS-SN: {app_mag_asassn_amplitude}')

    print('----------------------Gaia dr3--------------------------')

    app_mag_gaia_g_mean = ufloat(14.9546, 0.02)  # apparent magnitude
    app_mag_gaia_bp_mean = ufloat(15.5650, 0.02)  # apparent magnitude
    app_mag_gaia_rp_mean = ufloat(14.0896, 0.02)  # apparent magnitude

    color_gaia_bp_rp = app_mag_gaia_bp_mean - app_mag_gaia_rp_mean
    print(f'Color Gaia BP - RP: {color_gaia_bp_rp}')

    print('----------------------2MASS--------------------------')

    mag_2mass_j = ufloat(12.734, 0.027)
    mag_2mass_h = ufloat(11.977, 0.030)
    mag_2mass_k = ufloat(11.831, 0.029)

    color_2mass_j_k = mag_2mass_j - mag_2mass_k
    color_2mass_j_h = mag_2mass_j - mag_2mass_h
    print(f'Color 2MASS J - K: {color_2mass_j_k}')
    print(f'Color 2MASS J - H: {color_2mass_j_h}')

    print('----------------------AllWISE-------------------------')

    mag_allwise_w1 = ufloat(11.659, 0.024)
    mag_allwise_w2 = ufloat(11.658, 0.022)
    mag_allwise_w3 = ufloat(11.523, 0.192)
    mag_allwise_w4 = ufloat(9.340, 0.000000001)

    color_allwise_w1_w2 = mag_allwise_w1 - mag_allwise_w2
    color_allwise_w2_w3 = mag_allwise_w2 - mag_allwise_w3
    color_allwise_w3_w4 = mag_allwise_w3 - mag_allwise_w4
    print(f'Color AllWISE W1 - W2: {color_allwise_w1_w2}')
    print(f'Color AllWISE W2 - W3: {color_allwise_w2_w3}')
    print(f'Color AllWISE W3 - W4: {color_allwise_w3_w4}')

    star4 = VariableStarType(
        name="Star 4",
        type=None,
        period=period,
        temperature=temp_eff,
        bp_rp=color_gaia_bp_rp,
        j_k=color_2mass_j_k,
        j_h=color_2mass_j_h,
        w1_w2=color_allwise_w1_w2,
        w2_w3=color_allwise_w2_w3,
        w3_w4=color_allwise_w3_w4
    )
    return star4

class ExampleStar:
    def __init__(self, name, m_k, bp_rp, period, spect_type, paralax=None, dist=None):
        self.name = name
        self.m_k = m_k
        self.bp_rp = bp_rp
        self.period = period
        self.spect_type = spect_type
        self.paralax = paralax
        self.dist = dist

star4_full = star_4()
star4 = ExampleStar(
    name="Star 4",
    m_k=11.831,
    bp_rp=star4_full.bp_rp.n,  
    period=star4_full.period.n,
    spect_type='K',
    dist=[62440, 15286, 20063], # area, gaia, photo-geo, geo
    paralax=0.0067 
)

anara = ExampleStar(name="AN Ara", m_k=5.485, bp_rp=3.1965, period=153, spect_type='M1e', paralax=0.3276, dist=[0,0,1637,0])
l2pup = ExampleStar(name="L2 Pup", m_k=-1.97, bp_rp=3.5986, period=140.6, spect_type='M5IIIe-M6IIIe', paralax=17.7906)
rwcyg = ExampleStar(name="RW Cyg", m_k=0.48, bp_rp=4.0760, period=550, spect_type='M2-M4Ia-Iab', paralax=0.5766)
svuma = ExampleStar(name="SV UMa", m_k=6.963, bp_rp=1.3646, period=76, spect_type='G1Ibe-K3Iap', paralax=0.3073)
chicyg = ExampleStar(name="$\\chi$ Cyg", m_k=-2.09, bp_rp=7.3831, period=408.05, spect_type='S6,2e-S10,4e(MSe)', paralax=6.2686, dist=[0,0,131,0])
acher = ExampleStar(name="AC Her", m_k=5.075, bp_rp=0.9725, period=75.29, spect_type='F2pIb-K4e(C0,0)', paralax=0.5887, dist=[0,0,1403,0])
rvtau = ExampleStar(name="RV Tau", m_k=4.777, bp_rp=2.0693, period=78.731, spect_type='G2eIa-M2Ia', paralax=0.7603, dist=[0,0,1555,0])
    
def abs_mag(m_k, distance_pc=None, paralax=None):
    if distance_pc is None:
        distance_pc = 1000 / paralax
    abs_magnitude = m_k + 5 - 5 * np.log10(distance_pc)
    return abs_magnitude

stars = [star4, anara, l2pup, rwcyg, svuma, chicyg, acher, rvtau]
colors = [['magenta', 'purple', 'lightpink'], 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']
shapes = [['o', 'o', 'o'], 's', '^', '*', 'v', 'P', 'D', 'X']
zorders = [[10,9,8],5,5,5,6,7,5,5]


main_size = 100
example_size = 50

def draw_lum_per():
    img = Image.open(Path(__file__).parent.parent / 'output' / 'which_type' / 'lum_per_empty.jpg')
    w, h = img.size
    d = ImageDraw.Draw(img)
    d.text((250,10), "Luminosity-Period Diagram", fill=(0,0,0), font=ImageFont.truetype("arial.ttf", 50))

    figsize = (w / 300, h / 300)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_alpha(0)

    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


    for star, color, marker, zorder in zip(stars, colors, shapes, zorders):
        if star.name == "Star 4":
            ax.scatter(np.log10(star.period), abs_mag(star.m_k, distance_pc=star.dist[0]), color=color[0], marker=marker[0], zorder=zorder[0], label=f'{star.name}: $d_{{SMC}}$', s=main_size, edgecolors='black')
            print(f'Star 4 at SMC distance: M_K = {abs_mag(star.m_k, distance_pc=star.dist[0])}')
            ax.scatter(np.log10(star.period), abs_mag(star.m_k, distance_pc=star.dist[1]), color=color[1], marker=marker[1], zorder=zorder[1], label=f'{star.name}: $d_{{photo-geo}}$', s=main_size, edgecolors='black')
            ax.scatter(np.log10(star.period), abs_mag(star.m_k, distance_pc=star.dist[2]), color=color[2], marker=marker[2], zorder=zorder[2], label=f'{star.name}: $d_{{geo}}$', s=main_size, edgecolors='black')

        else:
            M_k = abs_mag(star.m_k, paralax=star.paralax)
            ax.scatter(np.log10(star.period), M_k, color=color, marker=marker, zorder=zorder, label=star.name, s=example_size, edgecolors='black')

    color = img.getpixel((0, 0))
    color_normalized = tuple(c/255 for c in color)
    ax.text(0.98, 0.064, f'log(P [days])', fontsize=9, color='black', va='bottom', ha='right', bbox=dict(facecolor=color_normalized, alpha=1, edgecolor='none'), transform=ax.transAxes)
    ax.text(0.03, 0.966, f'$M_K$ [Mag]', fontsize=9, color='black', va='top', ha='left', bbox=dict(facecolor=color_normalized, alpha=1, edgecolor='none'), transform=ax.transAxes)

    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Absolute Magnitude M_K')
    ax.invert_yaxis() 
    ax.set_xlim(-2, 3.54)
    ax.set_ylim(10, -12.2)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.02, 0.92), framealpha=0.6)
    plt.grid()
    plt.savefig(Path(__file__).parent.parent / 'output' / 'which_type' / 'lum_per_diagram_temp.png', transparent=True)

    img_temp = Image.open(Path(__file__).parent.parent / 'output' / 'which_type' / 'lum_per_diagram_temp.png')
    img.paste(img_temp, (0,0), img_temp)
    img.save(Path(__file__).parent.parent / 'output' / 'which_type' / 'lum_per_diagram.png',
                format='PNG',
                quality=95,
                subsampling=0,
                optimize=True)

def draw_hr_diagram():
    img = Image.open(Path(__file__).parent.parent / 'output' / 'which_type' / 'hr_empty.jpg')
    w, h = img.size
    d = ImageDraw.Draw(img)
    d.text((250,10), "H-R Diagram", fill=(0,0,0), font=ImageFont.truetype("arial.ttf", 50))

    figsize = (w / 300, h / 300)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_alpha(0)


    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for star, color, marker, zorder in zip(stars, colors, shapes, zorders):
        if star.name == "Star 4":
            ax.scatter(star.bp_rp, abs_mag(star.m_k, distance_pc=star.dist[0]), color=color[0], marker=marker[0], zorder=zorder[0], label=f'{star.name}: $d_{{SMC}}$', s=main_size, edgecolors='black')
            ax.scatter(star.bp_rp, abs_mag(star.m_k, distance_pc=star.dist[1]), color=color[1], marker=marker[1], zorder=zorder[1], label=f'{star.name}: $d_{{photo-geo}}$', s=main_size, edgecolors='black')
            ax.scatter(star.bp_rp, abs_mag(star.m_k, distance_pc=star.dist[2]), color=color[2], marker=marker[2], zorder=zorder[2], label=f'{star.name}: $d_{{geo}}$', s=main_size, edgecolors='black')
        else:
            M_k = abs_mag(star.m_k, paralax=star.paralax)
            ax.scatter(star.bp_rp, M_k, color=color, marker=marker, zorder=zorder, label=star.name, s=example_size, edgecolors='black')

    color = img.getpixel((0, 0))
    color_normalized = tuple(c/255 for c in color)
    ax.text(0.98, 0.07, f'BP-RP [Mag]', fontsize=9, color='black', va='bottom', ha='right', bbox=dict(facecolor=color_normalized, alpha=1, edgecolor='none'), transform=ax.transAxes)
    ax.text(0.03, 0.943, f'$M_K$ [Mag]', fontsize=9, color='black', va='top', ha='left', bbox=dict(facecolor=color_normalized, alpha=1, edgecolor='none'), transform=ax.transAxes)

    ax.set_xlabel('Color (BP - RP)')
    ax.set_ylabel('Absolute Magnitude M_K')
    ax.set_title('H-R Diagram')
    ax.invert_yaxis()
    ax.set_xlim(-0.7, 9.1)
    ax.set_ylim(10, -12.7)
    ax.legend(fontsize='small', loc='center right', ncols=2, bbox_to_anchor=(1, 0.31), framealpha=0.6)
    plt.grid()
    plt.savefig(Path(__file__).parent.parent / 'output' / 'which_type' / 'hr_diagram_temp.png', transparent=True)

    img_temp = Image.open(Path(__file__).parent.parent / 'output' / 'which_type' / 'hr_diagram_temp.png')
    img.paste(img_temp, (0,0), img_temp)
    img.save(Path(__file__).parent.parent / 'output' / 'which_type' / 'hr_diagram.png',
                format='PNG',
                quality=95,
                subsampling=0,
                optimize=True)

def draw_spectrums():
    plt.figure(figsize=(10,6))
    for star, color in zip(stars, colors):
        if star.name == "Star 4":
            folder_path = Path(__file__).parent.parent / 'data' / 'gaia_data' / 'star4'
            color = 'magenta'
            line_style = '-'
            line_width = 3
            z_order = 10
        elif star.name == "$\\chi$ Cyg":
            folder_path = Path(__file__).parent.parent / 'output' / 'which_type' / 'Chi Cyg'
            line_style = '--'
            line_width = 2
            z_order = 5
        else:
            folder_path = Path(__file__).parent.parent / 'output' / 'which_type' / star.name
            line_style = '--'
            line_width = 2
            z_order = 5
    
        cvs_file = [f for f in folder_path.glob("*.csv") if "XP_S" in f.name][0]
        print(f'Loading spectrum for {star.name} from {cvs_file}')
        #load cvs file
        df = pd.read_csv(cvs_file)
        wavelength = df['wavelength'].values
        flux = df['flux'].values

        flux_normized = flux / np.max(flux)

        plt.plot(wavelength, flux_normized, label=star.name, color=color, linestyle=line_style, linewidth=line_width, zorder=z_order)

    plt.xlabel('Wavelength [nm]', fontsize=14)
    plt.ylabel(f'Flux/Flux$_{{max}}$', fontsize=14)
    plt.title('Spectrum Comparison Diagram', fontsize=18)
    plt.legend(fontsize=12)
    plt.xlim(300, 1100)
    plt.grid()
    plt.savefig(Path(__file__).parent.parent / 'output' / 'which_type' / 'spectrums.png', dpi=300)
    plt.close()

    
draw_lum_per()
draw_hr_diagram()
draw_spectrums()

