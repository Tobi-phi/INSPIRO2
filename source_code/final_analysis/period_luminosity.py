import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
# import dustmaps

from get_data_surveys import get_data_assasn, get_data_gaia
from lomb_scargle_periodogram import compute_lomb_scargle
sys.path.append(str(Path(__file__).resolve().parent.parent)) 
from classes import star4, star9, star10, star12, star13, star14

### This file plots the star's location on the luminosity-period diagram

# apparent magnitude should be the average magnitude from light curves
# distance in parsecs from Vizier
# take extinction into account

def get_app_mag_from_light_curve(star):
    jd, mag, mag_err, camera = get_data_assasn(star)
    mag_avg_assasn = np.mean(mag)
    print(f"Star {star.id} Average Apparent Magnitude (ASAS-SN): {mag_avg_assasn}")

    jd_g, mag_g, jd_bp, mag_bp, jd_rp, mag_rp = get_data_gaia(star)
    mag_avg_g_gaia = np.mean(mag_g)
    mag_avg_bp_gaia = np.mean(mag_bp)
    mag_avg_rp_gaia = np.mean(mag_rp)
    print(f"Star {star.id} Average Apparent Magnitude (Gaia G): {mag_avg_g_gaia}")
    print(f"Star {star.id} Average Apparent Magnitude (Gaia BP): {mag_avg_bp_gaia}")
    print(f"Star {star.id} Average Apparent Magnitude (Gaia RP): {mag_avg_rp_gaia}")

    return mag_avg_assasn, mag_avg_g_gaia, mag_avg_bp_gaia, mag_avg_rp_gaia

# acquired period -> get luminosity/abs mag from luminosity-period diagram -> get distance and compare to literature



# get distance from literature -> get abs mag from app mag and distance -> get luminosity -> plot star on luminosity-period diagram 

def abs_mag_from_distance_and_app_mag(star):
    #app_mag_assasn, app_mag_g_gaia, app_mag_bp_gaia, app_mag_rp_gaia = get_app_mag_from_light_curve(star)
    app_mag_assasn = [11.831,11.831-0.029,11.831+0.029] 

    geo_dist = np.array([star.geo_dist[0], star.geo_dist[2], star.geo_dist[1]])
    photo_geo_dist = np.array([star.photo_geo_dist[0], star.photo_geo_dist[2], star.photo_geo_dist[1]])
    dist = star.dist
    area_dist = np.array(star.area_dist)

    extended_error_star4 = 5780
    area_dist_w_extended_error = np.array([area_dist[0], area_dist[2]+extended_error_star4, area_dist[1]-extended_error_star4])
    print('------------------------------------------------------------')
    print(f'error distance together: ±({extended_error_star4} + {area_dist[0]-area_dist[1]} = {extended_error_star4 + (area_dist[0]-area_dist[1])}) pc')
    print('------------------------------------------------------------')

    print(f"Star {star.id} Distances (pc):")
    print(f"Geometric Distance: {geo_dist} pc ->  {geo_dist[0]} (-{np.abs(geo_dist[0]-geo_dist[2]):.2f}/+{np.abs(geo_dist[1]-geo_dist[0]):.2f})")
    print(f"Photogeometric Distance: {photo_geo_dist} pc ->  {photo_geo_dist[0]} (-{np.abs(photo_geo_dist[0]-photo_geo_dist[2]):.2f}/+{np.abs(photo_geo_dist[1]-photo_geo_dist[0]):.2f})")
    print(f"Distance: {dist} pc")
    print(f"Area Distance: {area_dist_w_extended_error} pc ->  {area_dist_w_extended_error[0]} (-{np.abs(area_dist_w_extended_error[0]-area_dist_w_extended_error[2]):.2f}/+{np.abs(area_dist_w_extended_error[1]-area_dist_w_extended_error[0]):.2f})")

    print("------------------------------------------------------------")
    print(f"Star {star.id} Apparent Magnitude (ASAS-SN): {app_mag_assasn}")

    abs_mag_geo = app_mag_assasn - 5 * (np.log10(geo_dist) - 1)
    abs_mag_photo_geo = app_mag_assasn - 5 * (np.log10(photo_geo_dist) - 1)
    abs_mag_dist = app_mag_assasn - 5 * (np.log10(dist) - 1)
    abs_mag_area = app_mag_assasn - 5 * (np.log10(area_dist) - 1)
    abs_mag_area_w_extended_error = app_mag_assasn - 5 * (np.log10(area_dist_w_extended_error) - 1)

    print(f"----------------------------------------------------------------------------")
    print(f"Star {star.id} Absolute Magnitude (Geo Dist): {abs_mag_geo} -> {abs_mag_geo[0]} (-{np.abs(abs_mag_geo[1]-abs_mag_geo[0]):.2f}/+{np.abs(abs_mag_geo[0]-abs_mag_geo[2]):.2f})")
    print(f"Star {star.id} Absolute Magnitude (Photo Geo Dist): {abs_mag_photo_geo} -> {abs_mag_photo_geo[0]} (-{np.abs(abs_mag_photo_geo[1]-abs_mag_photo_geo[0]):.2f}/+{np.abs(abs_mag_photo_geo[0]-abs_mag_photo_geo[2]):.2f})")
    print(f"Star {star.id} Absolute Magnitude (Dist): {abs_mag_dist}")
    print(f"Star {star.id} Absolute Magnitude (Area Dist): {abs_mag_area}")
    print(f"Star {star.id} Absolute Magnitude (Area Dist with Extended Error): {abs_mag_area_w_extended_error} -> {abs_mag_area_w_extended_error[0]} (-{np.abs(abs_mag_area_w_extended_error[1]-abs_mag_area_w_extended_error[0]):.2f}/+{np.abs(abs_mag_area_w_extended_error[0]-abs_mag_area_w_extended_error[2]):.2f})")
    print(f"----------------------------------------------------------------------------")
    
    return abs_mag_geo, abs_mag_photo_geo, abs_mag_dist, abs_mag_area, abs_mag_area_w_extended_error

def luminosity_from_abs_mag(abs_mag):
    M_sun = 4.83  # Absolute magnitude of the Sun
    luminosity = 10 ** ((M_sun - abs_mag) / 2.5)
    return luminosity



def draw_onto_luminosity_period_diagram(star, img_path=Path(__file__).parent.parent / 'output' / 'which_type' / 'lum_per_empty.jpg'):
    def draw_dashed_line(draw, start, end, dash=10, gap=5, dashed=False, **kwargs):
        if dashed:
            x0, y0 = start
            x1, y1 = end
            dx = x1 - x0
            dy = y1 - y0
            length = (dx**2 + dy**2)**0.5
            ux, uy = dx/length, dy/length
            pos = 0
            while pos < length:
                x_start = x0 + ux * pos
                y_start = y0 + uy * pos
                pos += dash
                x_end = x0 + ux * min(pos, length)
                y_end = y0 + uy * min(pos, length)


                draw.line([(x_start, y_start), (x_end, y_end)], **kwargs)
                pos += gap
        else:
            draw.line([start, end], fill='black', width=10)
            draw.line([start, end], **kwargs)


    img_pl = Image.open(img_path)
    width, height = img_pl.size
    print(f'Image size: {img_pl.size}')
    draw = ImageDraw.Draw(img_pl)
    
    

    abs_mag_geo, abs_mag_photo_geo, abs_mag_dist, abs_mag_area, abs_mag_area_w_extended_error = abs_mag_from_distance_and_app_mag(star)

    abs_mag_area_value_smc = abs_mag_area_w_extended_error[0]
    abs_mag_area_minus_smc = abs(abs_mag_area_w_extended_error[1] - abs_mag_area_value_smc)
    abs_mag_area_plus_smc = abs(abs_mag_area_w_extended_error[2] - abs_mag_area_value_smc)

    abs_mag_geo_value = abs_mag_geo[0]
    abs_mag_geo_minus = abs(abs_mag_geo[1] - abs_mag_geo_value)
    abs_mag_geo_plus = abs(abs_mag_geo[2] - abs_mag_geo_value)

    abs_mag_photo_geo_value = abs_mag_photo_geo[0]
    abs_mag_photo_geo_minus = abs(abs_mag_photo_geo[1] - abs_mag_photo_geo_value)
    abs_mag_photo_geo_plus = abs(abs_mag_photo_geo[2] - abs_mag_photo_geo_value)

    



    jd_asassn, mag_asassn, mag_err_asassn, camera_asassn = get_data_assasn(star)
    jd_g_gaia, mag_g_gaia, jd_bp_gaia, mag_bp_gaia, jd_rp_gaia, mag_rp_gaia = get_data_gaia(star)
    freq, power, best_frequency, best_period, second_frequency, second_period, min_indices, min_times, second_min_indicies, second_min_times = compute_lomb_scargle(jd_asassn, mag_asassn, mag_err_asassn, star)
    
    
    

    figsize = (width / 300, height / 300)  # size in inches for 300 dpi
    fig, ax = plt.subplots(figsize=figsize, dpi=300) 
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    
    text_per = f'{star.star}:$P_2$ = ({second_period[0]:.1f} $\\pm$ {second_period[1]:.2f}) days'
    x_log10_period = 1275
    color = 'rebeccapurple'
    draw_dashed_line(draw, (x_log10_period, 55), (x_log10_period, 750), dash=10, fill=color, gap=10, width=4)
    ax.text(0.2, 0.95, text_per, fontsize=12, color=color, va='top', ha='left') 

    text_Msmc = f'$M_K(d_{{SMC}})$ = ${abs_mag_area_value_smc:.2f}^{{+{abs_mag_area_plus_smc:.2f}}}_{{-{abs_mag_area_minus_smc:.2f}}}$'
    y_abs_mag = 175
    color = 'magenta'
    draw_dashed_line(draw, (25, y_abs_mag), (1450, y_abs_mag), dash=10, fill=color, gap=10, width=4)
    ax.text(0.04, 0.86, text_Msmc, fontsize=12, color=color, va='top', ha='left') 
    
    text_Mgeo = f'$M_K(d_{{geo}})$ = ${abs_mag_geo_value:.2f}^{{+{abs_mag_geo_plus:.2f}}}_{{-{abs_mag_geo_minus:.2f}}}$'
    y_abs_mag = 257
    color = 'lightpink'
    draw_dashed_line(draw, (25, y_abs_mag), (1450, y_abs_mag), dash=10, fill=color, gap=10, width=4)
    ax.text(0.04, 0.755, text_Mgeo, fontsize=12, color=color, va='top', ha='left')

    text_Mphoto_geo = f'$M_K(d_{{photo-geo}})$ = ${abs_mag_photo_geo_value:.2f}^{{+{abs_mag_photo_geo_plus:.2f}}}_{{-{abs_mag_photo_geo_minus:.2f}}}$'
    y_abs_mag = 270
    color = 'purple'
    draw_dashed_line(draw, (25, y_abs_mag), (1450, y_abs_mag), dash=10, fill=color, gap=10, width=4)
    ax.text(0.04, 0.61, text_Mphoto_geo, fontsize=12, color=color, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    color = img_pl.getpixel((0, 0))
    color_normalized = tuple(c/255 for c in color)
    print(f'Background color: {color}')


    
    ax.text(0.98, 0.06, f'$\\log_{{10}}\\frac{{P}}{{\\text{{days}}}}$', fontsize=9, color='black', va='bottom', ha='right', bbox=dict(facecolor=color_normalized, alpha=1, edgecolor='none'), transform=ax.transAxes)
    ax.text(0.03, 0.962, f'$M_K$ [Mag]', fontsize=9, color='black', va='top', ha='left', bbox=dict(facecolor=color_normalized, alpha=1, edgecolor='none'), transform=ax.transAxes)
    fig.patch.set_alpha(0)  # make background transparent
    temp_path = Path(__file__).parent.parent / 'output' / 'lum_per_diagram' / 'temp_text.png'
    fig.savefig(temp_path, dpi=300, transparent=True)
    plt.close(fig)

    text_img = Image.open(temp_path)
    img_pl.paste(text_img, (0, 0), text_img)  # use text_img as mask for transparency

    var_type_list_img = Image.open(Path(__file__).parent.parent / 'output' / 'lum_per_diagram' / f'var_types.jpg')
    var_type_img_width, var_type_img_height = var_type_list_img.size
    scale_factor = 1.5
    var_type_list_img = var_type_list_img.resize((int(var_type_img_width * scale_factor), int(var_type_img_height * scale_factor)))
    img_pl.paste(var_type_list_img, (1330, 0))

    img_pl.save(Path(__file__).parent.parent / 'output' / 'lum_per_diagram' / f'lum_per_diagram_star_{star.id}_advanced.png',
                format='PNG',
                quality=95,
                subsampling=0,
                optimize=True)

    # OLD 
    # font = ImageFont.truetype("arial.ttf", 25)
    # draw.text((100, 60), text,
    #            fill=star.color, font=font)

    # img_pl.save(Path(__file__).parent.parent / 'output' / 'lum_per_diagram' / f'lum_per_diagram_star_{star.id}.jpg',
    #             format='JPEG',
    #             quality=95,
    #             subsampling=0,
    #             optimize=True)

    
# draw_onto_luminosity_period_diagram(star4)
# draw_onto_luminosity_period_diagram(star9)

