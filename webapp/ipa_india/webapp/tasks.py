#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 06:44:31 2021

@author: lucadelu
@author: spareeth

"""
import os
import sys
import subprocess
import statistics
import shutil
import calendar
from django.conf import settings
from celery import shared_task, current_task
from .functions import render_prod_html
from .functions import render_pdf_html
from .functions import get_concat_multi_resize
from .functions import render_pdf
from .functions import send_mail_attach
from .models import Area
from PIL import Image
import math
os.environ.update({'GRASSBIN': settings.GRASS_BIN})
##export LD_LIBRARY_PATH=$(grass78 --config path)/lib
from grass_session import TmpSession
from grass.pygrass.modules.shortcuts import general as g
from grass.pygrass.modules.shortcuts import raster as r
from grass.pygrass.modules.shortcuts import display as d
from grass.pygrass.modules.shortcuts import vector as v
from grass.pygrass.gis import *
import grass.script as grass
import grass.script.setup as gsetup

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as colors
from sklearn.linear_model import Ridge
import geopandas as gdf
import pandas as pd
import pymannkendall as mk



@shared_task(bind=True)
def report_basin(self, area, start_month, end_month, precip, et,lcc,current_user):
    print('start_month:')
    print(start_month)

    print('end_month:')
    print(end_month)
    print('PCP:')
    print(precip)
    print('ET:')
    print(et)
    print('LCC:')
    print(lcc)

    print('current_user:')
    print(current_user.email)


    lulc_available = {
        "LULC_250k_1516_epsg4326":"2015-16",
        "LULC_250k_1617_epsg4326":"2016-17" ,
        "LULC_250k_1718_epsg4326":"2017-18" ,
        "LULC_250k_2223_epsg4326":"2022-23"
    }
    selected_lcc_year=lulc_available[lcc]


    total_steps = 100 
    progress = 0

    self.update_state(state='PROGRESS', meta={'current': progress, 'total': total_steps})


    start_month_name = calendar.month_name[int(start_month)]
    end_month_name = calendar.month_name[int(end_month)]


    seasons_timerange = []

    start_yr = '2018'
    end_yr = '2023' if int(start_month) <= int(end_month) else '2022'

    # Loop through each year in the range 2018 to 2023
    for year in range(int(start_yr),int(end_yr)+1):
        months_range = []

        if int(start_month) > int(end_month):
            # First, add the months from start_month to December of the current year
            for month in range(int(start_month), 13):  # 13 because we want to include month 12 (December)

                months_range.append(f"{year}_{month:02d}")
            # Then, add the months from January to end_month of the next year
            for month in range(1, int(end_month) + 1):
                months_range.append(f"{year + 1}_{month:02d}")
        else:
            # If start_month <= end_month, just add the months within the same year
            for month in range(int(start_month), int(end_month) + 1):
                months_range.append(f"{year}_{month:02d}")
        seasons_timerange.append(months_range)
    

    print("months_timerange:")
    print(seasons_timerange)



    timerange = range(int(start_yr),int(end_yr)+1)
    print("timerange:")
    print(timerange)

    


    years = list(timerange)


    years_str = [str(s) for s in years]
    print("years_str")
    print(years_str)
    
    jobid = self.request.id
    LC_ESA = os.path.join(settings.DATA_DIR, 'worldcover_ESA')
    #create a new directory for each task as set names
    newdir = os.path.join(settings.MEDIA_ROOT, jobid)
    print("newdir")
    print(newdir)
    os.mkdir(newdir)
    #GRASS variables
    os.environ.update(dict(GRASS_COMPRESS_NULLS='1',
                           GRASS_COMPRESSOR='ZSTD',
                           GRASS_OVERWRITE='1'))
    # create a GRASS Session instance with a new location and mapset
    user = TmpSession()
    #user.open(gisdb=settings.GRASS_DB, location='job{}'.format(jobid),
    #               create_opts='EPSG:4326')
    #user.open(gisdb=settings.GRASS_DB, location='wagen',
                   #mapset='job{}'.format(jobid), create_opts='EPSG:4326')
    user.open(gisdb=settings.GRASS_DB, location='wagen',
                   mapset='job{}'.format(jobid), create_opts='')
    #gisdb=settings.GRASS_DB
    #location='wagen'
    #mapset='job{}'.format(jobid)
    #session = gsetup.init(gisdb, location, mapset)

    from grass.pygrass.raster import RasterRow
    from grass.pygrass.gis.region import Region
    from grass.pygrass.raster import raster2numpy

    #get the area and create a new GRASS vector for this
    #from grass.pygrass.vector import VectorTopo
    #from grass.pygrass.vector import geometry as geo 
    myarea = Area.objects.get(id=area)
    #centroid = geo.Point(myarea.geom.centroid.x, myarea.geom.centroid.y)
    #bound = geo.Line([myarea[0][0]])
    #vectname = "{na}_{job}".format(na=myarea.name, job=jobid.replace("-", "_"))
    vectname = "{na}_{job}".format(na=myarea.name.replace(' ', '').replace("-", "_").replace("'", "_").replace("ô", "_").replace("&", "and").replace("(", "_").replace(")", "_"), job=jobid.replace("-", "_"))

    buffered_vectname = f"{vectname}_buffered"


    print("area of Interest: ")
    print(myarea.name)

    print("AOI State: ")
    state=myarea.state
    print(state)

    print("jobid: ")
    print(jobid)
        #new = VectorTopo(vectname)
    #new.open('w')
    #area = geo.Area(boundary=bound, centroid=centroid)
    #new.write(area)
    #new.close()
    v.in_ogr(input="PG:dbname={db} host={add} port={po} user={us} password={pwd}".format(db=settings.DATABASES['default']['NAME'], add=settings.DATABASES['default']['HOST'], po=settings.DATABASES['default']['PORT'],  us=settings.DATABASES['default']['USER'], pwd=settings.DATABASES['default']['PASSWORD']), output=vectname, where="id={}".format(area))

    out1 = os.path.join(newdir, "bound.gpkg")
    out2 = os.path.join(newdir, "bound.geojson")
    canals_aoi = os.path.join(newdir, "canals.geojson")

    v.out_ogr(input=vectname, output=out1)
    v.out_ogr(input=vectname, output=out2, format='GeoJSON')


    


    india_canal = os.path.join(settings.BASE_DIR, 'staticData/India_Canals.geojson')
    v.import_(input=india_canal, output='india_canal_layer', overwrite=True)


    


    cent = grass.parse_command('v.out.ascii', input=vectname, type='centroid', format='point', separator='comma')
    centkeys = list(cent.keys())
    centlist = [item for items in centkeys for item in items.split(",")]
    centX = float(centlist[0])
    centY = float(centlist[1])
    # execute some command inside PERMANENT
    # Add the data mapsets in search path
    g.mapsets(mapset="etg_etb_ind_monthly,data_annual,data_monthly,grace,cmip_ssp245,cmip_ssp585,pcp_era5,pcp_gpm,pcp_gsmap,imd_daily,nrsc_lulc,wapor3_ea_dekadal,wapor3_ia_dekadal,wapor3_ta_dekadal", operation="add")


    
    g.region(vector=vectname, res=0.003)
    bbox = grass.parse_command('g.region', flags='pg')
    df = gdf.read_file(out1)
    # compute area in ha of the studyarea
    grass.run_command('v.to.db', map=vectname, option='area', type='boundary', units='kilometers', columns='area')
    area_col = grass.parse_command('v.univar', map=vectname, column='area', flags=('g'))
    studyarea = int(float(area_col['min']))

    if studyarea < 100:
        buffer_degrees = 0.018  # 2km buffer (0.018 degree)
    elif studyarea < 500:
        buffer_degrees = 0.036  # 4km buffer
    elif studyarea < 1000:
        buffer_degrees = 0.054   # 6km buffer
    elif studyarea < 2000:
        buffer_degrees = 0.072  # 8km buffer
    else:
        buffer_degrees = 0.09  # 10km buffer
    print(f"Calculated buffer distance: {buffer_degrees} degrees")



    # buffer_km = studyarea / 100
    # buffer_degrees = buffer_km / 111
    # print(f"Calculated buffer distance: {buffer_degrees} degrees")



    buffered_aoi = os.path.join(newdir, "buffered_bound.geojson")
    v.buffer(input=vectname, output=buffered_vectname, distance=buffer_degrees)

    v.out_ogr(input=buffered_vectname, output=buffered_aoi, format='GeoJSON')


    v.overlay(ainput='india_canal_layer', binput=buffered_vectname, output='aoi_canal', operator='and')
    v.out_ogr(input='aoi_canal', output=canals_aoi, format='GeoJSON')




    # Extract the extents
    #spatial_extent=(28.9134,32.6882,29.9117,31.5939)
    west = round(float(bbox['w']), 2)
    east = round(float(bbox['e']), 2)
    north = round(float(bbox['n']), 2)
    south = round(float(bbox['s']), 2)
    spatial_extent=(float(bbox['w']),float(bbox['e']),float(bbox['s']),float(bbox['n']))
    ##Correcting Region() manually for the raster2numpy to work
    # print("hallo again")
    #reg = Region().from_vect(vectname)
    reg = Region()
    reg.north = float(bbox['n'])
    reg.south = float(bbox['s'])
    reg.west = float(bbox['w'])
    reg.east = float(bbox['e'])
    reg.nsres = float(bbox['nsres'])
    reg.ewres = float(bbox['ewres'])
    reg.rows = int(bbox['rows'])
    reg.cols = int(bbox['cols'])
    reg.write()
    reg.set_raster_region()
    
    # Adding mask to study area
    # remaining figures/stats only on masked area
    r.mask(vector=vectname)
    # Extract the DEM stats
    grass.mapcalc('{r} = int({a} * 1.0)'.format(r=f'dem_studyarea', a=f'dem_alos'))    
    dem_stats = grass.parse_command('r.univar', map=f'dem_studyarea', flags='eg', percentile='2,98')
    p2 = float(dem_stats['percentile_2'])
    p98 = float(dem_stats['percentile_98'])
    dem_min = int(float(dem_stats['percentile_2']))
    dem_max = int(float(dem_stats['percentile_98']))
    
    ### Figure 1 - study area ###
    demtif = os.path.join(newdir, "DEM.tif")
    r.out_gdal(input='dem_studyarea', output=demtif)
    dem=raster2numpy("dem_studyarea", mapset='job{}'.format(jobid))
    dem = np.ma.masked_where(dem == -2147483648, dem)
    fig1 = os.path.join(newdir, "fig1.png")
    fig, ax = plt.subplots(figsize = (12,8))
    plt.imshow(dem, cmap='terrain', vmin=np.nanmin(dem), vmax=np.nanmax(dem), extent=spatial_extent)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    #df.boundary.plot(ax=ax, facecolor='none', edgecolor='k', label='Boundary');
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
    x, y, arrow_length = 1.1, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=18, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.colorbar(shrink=0.50, label='Elevation[meters]')
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=10)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=10)
    plt.title('Study area', fontsize=10)
    plt.savefig(fig1, bbox_inches='tight',pad_inches = 0, dpi=100)

    progress += 10
    self.update_state(state='PROGRESS', meta={'current': progress, 'total': total_steps})

    ### Prepare Landcover for study area


    grass.mapcalc('{r} = {a}'.format(r=f'LC_studyarea1', a=lcc))
    print('Using NRSC LCC map')

    print('LC details starts here:')
   
    LCcsv = os.path.join(newdir, "LC.csv")
    LC_stats = grass.parse_command('r.stats',flags='napl',separator='comma',input='LC_studyarea1',output=LCcsv)
    print("LC_stats: ")
    print(LC_stats)
    LCdf = pd.read_csv(LCcsv, header=None, sep=',')
    LCdf_filt = LCdf.loc[LCdf[2] > 2500]
    LC_code = LCdf_filt[0].tolist()
    print(LC_code)
    LC_name = LCdf_filt[1].tolist()
    print(LC_name)
    LC_area = LCdf_filt[2].tolist()
    print(LC_area)
    LC_perc_str = LCdf_filt[3].tolist()
    print(LC_perc_str)

    # Update LC_study area removing the land cover types which are less than 10 pixels
    LCdf_filt1 = LCdf.loc[LCdf[2] < 2500]
    LC_code1 = LCdf_filt1[0].tolist()
    LC_code1_str=[str(item) for item in LC_code1]
    LC_code_mask=" ".join(LC_code1_str)
    print(f"LC to be masked are {LC_code_mask}")

    if not LC_code1_str:
            print('No classes to be ignored')
            grass.mapcalc('{r} = {a}'.format(r=f'LC_studyarea', a=f'LC_studyarea1'))
    else:
            r.mask(raster="LC_studyarea1", maskcats=LC_code_mask, flags="i")
            grass.mapcalc('{r} = {a}'.format(r=f'LC_studyarea', a=f'LC_studyarea1'))
    r.mask(vector=vectname)
    LC_perc = [item.replace("%", "") for item in LC_perc_str]
    LC_perc_flt = [round(float(item), 1) for item in LC_perc]
    print(LC_perc_flt)
    LC_area_ha = [int(round(x/10000)) for x in LC_area]
    print(LC_area_ha)
    LC_area_sqkm = [int(round(x/100)) for x in LC_area_ha]
    print(LC_area_sqkm)
    ### Extract color codes from Landcover ###
    LC_color = grass.parse_command('r.what.color',input='LC_studyarea',value=LC_code,format='#%02x%02x%02x')
    LC_color_keys = list(LC_color.keys())
    LC_color_comma = [item.replace(": ", ",") for item in LC_color_keys]
    y = [item for items in LC_color_comma for item in items.split(",")]
    LC_color_hex = y[1::2]

    # Extract color codes second time for the pie chart of Landcover
    #LC_color1 = grass.parse_command('r.what.color',input='LC_studyarea',value=LC_code,format='#%02x%02x%02x')
    #LC_color1_keys = list(LC_color1.keys())
    #LC_color1_comma = [item.replace(": ", ",") for item in #LC_color1_keys]
    #y1 = [item for items in LC_color1_comma for item in items.split(",")]
    #LC_color1_hex = y1[1::2]
    
    r.mask(flags="r")
    g.region(flags='d')

    g.region(vector=buffered_vectname, res=0.003)
    r.mask(vector=buffered_vectname)
    grass.mapcalc('{r} = {a}'.format(r=f'LC_studyarea_buffered', a=lcc))

    bbox_buffer = grass.parse_command('g.region', flags='pg')

    buffer_spatial_extent=(float(bbox_buffer['w']),float(bbox_buffer['e']),float(bbox_buffer['s']),float(bbox_buffer['n']))



    ### Figure 2 - Landcover ###
    lc = os.path.join(newdir, "lc.png")
    lcpie = os.path.join(newdir, "pie.png")
    leg = os.path.join(newdir, "leg.png")
    
    lctif = os.path.join(newdir, "LC.tif")
    r.out_gdal(input='LC_studyarea_buffered', output=lctif)
    
    LC=raster2numpy("LC_studyarea_buffered", mapset='job{}'.format(jobid))
    LC = np.ma.masked_where(LC == -2147483648, LC)
    fig, ax = plt.subplots(figsize = (12,8))
    cmap = ListedColormap(LC_color_hex)
    print(LC_color_hex)
    print(LC_code)
    LC_code.insert(0, -1)
    LC_code1 = [x+1 for x in LC_code]
    norm = BoundaryNorm(LC_code1, cmap.N)
    plt.imshow(LC, cmap=cmap, extent=buffer_spatial_extent, norm=norm, interpolation='nearest', resample=True)
    #plt.imshow(LC, cmap=cmap, extent=spatial_extent, norm=norm)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k',linewidth=2)
    gdf.read_file(canals_aoi).plot(ax=ax, color='blue', linewidth=1.5)
    x, y, arrow_length = 1.01, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
    plt.title('Major Land cover types', fontsize=12)
    plt.savefig(lc, bbox_inches='tight',pad_inches = 0, dpi=100)

    r.mask(flags="r")
    g.region(vector=vectname, res=0.003)

    r.mask(vector=vectname)


    progress += 10
    self.update_state(state='PROGRESS', meta={'current': progress, 'total': total_steps})


# LC comparision chart
    available_nrsc_lcc=['LULC_250k_1516_epsg4326','LULC_250k_1617_epsg4326','LULC_250k_1718_epsg4326','LULC_250k_2223_epsg4326']
    LC_compare_stats = {}
    for i in available_nrsc_lcc:
           yr = i.split("_")[2]
           lcc_c='LC_compare'+yr
           grass.mapcalc('{r} = {a}'.format(r=lcc_c, a=i))
           LC_c_stats = grass.parse_command('r.stats',flags='napl',separator='comma',input=lcc_c)
           LC_code = []
           LC_name = []
           LC_area = []

           # Process the statistics and split them into the desired components
           for key in LC_c_stats.keys():
                parts = key.split(',')
                LC_code.append(int(parts[0])) 
                LC_name.append(parts[1])     
                LC_area.append(round(float(parts[2])/ 1000000 ,0)) 
        
           # Add the processed data to the dictionary for the current year
           LC_compare_stats[yr] = {
                "LC_code": LC_code,
                "LC_name": LC_name,
                "LC_area": LC_area
            }



#     print("LC_compare_state",LC_compare_stats)
    common_LC_code = set(LC_compare_stats['1516']['LC_code'])

    for year in ['1617', '1718', '2223']:
        common_LC_code.intersection_update(set(LC_compare_stats[year]['LC_code']))

    common_LC_code = {code for code in common_LC_code if 2 <= code <= 7}
    common_LC_code = sorted(common_LC_code)
    filtered_data = {year: [] for year in ['1516', '1617', '1718', '2223']}
    filtered_LC_name = []  # To store corresponding LC names

    for year in ['1516', '1617', '1718', '2223']:
        for lc_code in common_LC_code:
                # Get the index of the LC_code in the current year
                lc_index = LC_compare_stats[year]['LC_code'].index(lc_code)
                # Append the corresponding LC_area and LC_name to the filtered data
                filtered_data[year].append(LC_compare_stats[year]['LC_area'][lc_index])
        
        # Only add names once (for any year), as LC_name should be consistent across years
        if not filtered_LC_name:
                filtered_LC_name = [LC_compare_stats['1516']['LC_name'][LC_compare_stats['1516']['LC_code'].index(lc_code)] for lc_code in common_LC_code]

        # Create the DataFrame using the filtered common LC codes and their areas
    df_lc_compare = pd.DataFrame({
        'LC_name': filtered_LC_name,
        '2015-16': filtered_data['1516'],
        '2016-17': filtered_data['1617'],
        '2017-18': filtered_data['1718'],
        '2022-23': filtered_data['2223']
        }, index=filtered_LC_name)
    
    lc_comp=os.path.join(newdir, "lc_comp.png")
    fig, ax = plt.subplots(figsize = (8,4))
    df_lc_compare.plot.bar(y = ['2015-16','2016-17','2017-18','2022-23'], rot = 40, ax = ax, color=['seagreen', 'skyblue', 'orange', 'purple'])
    #ax.invert_yaxis()
#     ax.set_title('Crop classes from all available NRSC lcc')
    ax.set_ylabel('Area (sq km)')
    plt.savefig(lc_comp, bbox_inches='tight',pad_inches = 0.1, dpi=100)
    


    ### LC Pie Chart ###
    y = np.array(LC_perc_flt)
    x = LC_name
    LC_colors = LC_color_hex
    fig, ax = plt.subplots(figsize = (4,4))
    patches, texts = plt.pie(y, colors=LC_colors, startangle=90, radius=1.2, shadow = True)
    plt.savefig(lcpie, bbox_inches='tight',pad_inches = 0, dpi=100)
    labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(x, y)]
    sort_legend = True
    if sort_legend:
            patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                            key=lambda x: x[2],
                            reverse=True))
    fig, ax = plt.subplots(figsize = (4,4))
    ax.axis('off')
    #plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=10)
    plt.legend(patches, labels, loc='center', fontsize=10, borderpad=0.2)
    plt.savefig(leg, bbox_inches='tight',pad_inches = 0, dpi=100)    
    ## Remove mask and set region to 500 m and reset mask
    r.mask(flags="r")
    g.region(vector=vectname, res=0.003)
    r.mask(vector=vectname)
    

    ## For IPA specific
    ## Create yearly Seasonal maps
    sn = 0
    for s in seasons_timerange:
        maps1 = ["wapor3_eta_" + i for i in s]
        maps2 = ["wapor3_etb_" + i for i in s]
        maps3 = ["wapor3_etg_" + i for i in s]
        maps4 = ["wapor3_npp_" + i for i in s]
        maps5 = ["wapor3_ea_" + i for i in s]
        maps6 = ["wapor3_ta_" + i for i in s]
        # maps7 = ["wapor3_ia_" + i for i in s]
        maps8 = ["pcpm_imd_" + i for i in s]
        maps9 = ["ssebop_etpm_" + i for i in s]
        sn = sn + 1
        yr = maps1[0].split("_")[2]
        outeta = et + '_eta_' + yr
        outetb = et + '_etb_' + yr
        outetg = et + '_etg_' + yr
        outnpp = et + '_npp_' + yr
        outta = et + '_ta_' + yr
        outea = et + '_ea_' + yr
        # outia = et + '_ia_' + yr
        outpcp = precip + '_pcp_' + yr
        outetp1 = 'ssebop' + '_etpm1_' + yr
        outetp = 'ssebop' + '_etpm_' + yr
        r.series(input=maps1, output=outeta, method='sum')
        r.series(input=maps2, output=outetb, method='sum')
        r.series(input=maps3, output=outetg, method='sum')
        r.series(input=maps4, output=outnpp, method='sum')
        r.series(input=maps5, output=outea, method='sum')
        r.series(input=maps6, output=outta, method='sum')
        # r.series(input=maps7, output=outia, method='sum')
        r.series(input=maps8, output=outpcp, method='sum')
        r.series(input=maps9, output=outetp1, method='sum')
        grass.mapcalc('{r} = {a} * 0.01'.format(r=outetp, a=outetp1))

        print(maps1)
        print(maps2)
        print(maps3)
        print(maps4)
        print(maps5)
        print(maps6)
        # print(maps7)
        print(maps8) 
        print(maps9)
 
 
    ## Average seasonal maps of ETa and Precip, over the years
    mapseta = [et + "_eta_" + s for s in years_str]
    mapsetb = [et + "_etb_" + s for s in years_str]
    mapsetg = [et + "_etg_" + s for s in years_str]
    mapsnpp = [et + "_npp_" + s for s in years_str]
    mapsta = [et + "_ta_" + s for s in years_str]
    mapsea = [et + "_ea_" + s for s in years_str]
#     mapsia = [et + "_ia_" + s for s in years_str]    
    mapspcp = [precip + '_pcp_' + s for s in years_str]
    mapsetp = ['ssebop' + '_etpm_' + s for s in years_str]
    r.series(input=mapseta, output='eta_mean', method='average')
    r.series(input=mapsetb, output='etb_mean', method='average')
    r.series(input=mapsetg, output='etg_mean', method='average')
    r.series(input=mapsnpp, output='npp_mean', method='average')
    r.series(input=mapsta, output='ta_mean', method='average')
    r.series(input=mapsea, output='ea_mean', method='average')
#     r.series(input=mapsia, output='ia_mean', method='average')
    r.series(input=mapspcp, output='pcp_mean', method='average')
    r.series(input=mapsetp, output='etr_mean', method='average')
    ## Monthly maps as png's
    ## Monthly maps of ETa
    months = range(1, 13)    
    for mm in months:
        n=str(mm).zfill(2)
        pattern='wapor3_eta_avg_'
        img=pattern+n
        print(img)
        plt1=img+'.png'
        plt2=os.path.join(newdir, plt1)
        rast=raster2numpy(img, mapset='etg_etb_ind_monthly')
        rast = np.ma.masked_where(rast == -2147483648, rast)
        fig, ax = plt.subplots(figsize = (12,8))
        #plt.imshow(rast, cmap='jet_r', vmin=10, vmax=np.nanmax(rast),extent=spatial_extent, interpolation='none', resample=False)
        plt.imshow(rast, cmap='RdYlGn', vmin=10, vmax=230,extent=spatial_extent, interpolation='none', resample=False)
        scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
        fig.gca().add_artist(scalebar)
        df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
        x, y, arrow_length = 0.9, 0.12, 0.08
        ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=6, headwidth=15),
                ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
        plt.colorbar(shrink=0.50, label='ETa [mm/month]', pad = 0.01, orientation='horizontal')
        #plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
        #plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12, labelpad=-4)
        # title='Monthly('+str(mm)+')'+(' ETa')
        title='Month - '+str(mm)
        plt.title(title, fontsize=18, pad=2)
        plt.savefig(plt2, bbox_inches='tight',pad_inches = 0, dpi=100)
    
    ## Monthly maps of PCP
    for mm in months:
        n=str(mm).zfill(2)
        pattern='pcpm_imd_avg_'
        img=pattern+n
        print(img)
        plt1=img+'.png'
        plt2=os.path.join(newdir, plt1)
        rast=raster2numpy(img, mapset='data_monthly')
        rast = np.ma.masked_where(rast == -2147483648, rast)
        fig, ax = plt.subplots(figsize = (12,8))
        #plt.imshow(ETa, cmap='jet_r', vmin=10, vmax=np.nanmax(ETa),extent=spatial_extent, interpolation='none', resample=False)
        plt.imshow(rast, cmap='Blues', vmin=10, vmax=400,extent=spatial_extent, interpolation='none', resample=False)
        scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
        fig.gca().add_artist(scalebar)
        df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
        x, y, arrow_length = 0.9, 0.12, 0.08
        ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=6, headwidth=15),
                ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
        plt.colorbar(shrink=0.50, label='P [mm/month]', pad = 0.01, orientation='horizontal')
        #plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
        #plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12, labelpad=-4)
        # title='Monthly('+str(mm)+')'+(' P')
        title='Month - '+str(mm)
        plt.title(title, fontsize=18, pad=2)
        plt.savefig(plt2, bbox_inches='tight',pad_inches = 0, dpi=100)
        

    progress += 10
    self.update_state(state='PROGRESS', meta={'current': progress, 'total': total_steps})

    with Image.open(lc) as img:
        plots_width, plots_height = img.size  # img.size returns a tuple (width, height)

    print(f"Image Width: {plots_width}")
    print(f"Image Height: {plots_height}")   



    monthly_pcp_plots=[]
    monthly_eta_plots=[]
    for m in range(1, 13):
          pcp_m_path=os.path.join(newdir, f"pcpm_imd_avg_{str(m).zfill(2)}.png")
          eta_m_path=os.path.join(newdir, f"wapor3_eta_avg_{str(m).zfill(2)}.png")
          monthly_pcp_plots.append(pcp_m_path)
          monthly_eta_plots.append(eta_m_path)

    images_pcp_m = [Image.open(img_path) for img_path in monthly_pcp_plots]
    images_eta_m = [Image.open(img_path) for img_path in monthly_eta_plots]
    webImg_style=''
    pdfImg_style=''
    pdfImg_style2=''

    if plots_width < plots_height:
        get_concat_multi_resize(im_list=images_pcp_m, n_rows=4).save(os.path.join(newdir, 'pcpm_imd_avg_all.png'))
        get_concat_multi_resize(im_list=images_eta_m, n_rows=4).save(os.path.join(newdir, 'wapor3_eta_avg_all.png'))
        webImg_style="height: 600px; width: 100%; text-align: center; border: 0.5px solid #e7e7e7;"
        pdfImg_style="height: 500px; width: 100%; text-align: center;"
        pdfImg_style2="height: 100%; width: auto; justify-self: center; margin-top: 10px;"

    else:
        get_concat_multi_resize(im_list=images_pcp_m, n_rows=6).save(os.path.join(newdir, 'pcpm_imd_avg_all.png'))
        get_concat_multi_resize(im_list=images_eta_m, n_rows=6).save(os.path.join(newdir, 'wapor3_eta_avg_all.png'))
        webImg_style='width: 100%; text-align: center; border: 0.5px solid #e7e7e7;'
        pdfImg_style="width: 100%; text-align: center;"
        pdfImg_style2="height: 100%; width: 100%; justify-self: center; margin-top: 10px;"
          

    ## FIGURE 3 - ETA plot ###
    etaplt = os.path.join(newdir, "eta.png")
    etatif = os.path.join(newdir, "ETa.tif")
    # r.out_gdal(input='eta_mean', output=etatif)
    r.out_gdal(input='eta_mean', output=etatif, nodata=-9999)

    ETa=raster2numpy('eta_mean', mapset='job{}'.format(jobid))
    ETa = np.ma.masked_invalid(ETa) 

    # ETa = np.ma.masked_where(ETa == -2147483648, ETa)
    fig, ax = plt.subplots(figsize = (12,8))
    plt.imshow(ETa, cmap='jet_r', vmin=10, vmax=np.nanmax(ETa),extent=spatial_extent, interpolation='none', resample=False)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
    x, y, arrow_length = 1.1, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.colorbar(shrink=0.50, label='ETa [mm/year]')
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
    plt.title('Seasonal ETa ', fontsize=12)
    plt.savefig(etaplt, bbox_inches='tight',pad_inches = 0, dpi=100)
    
    eta_basin = grass.parse_command('r.univar', map=f'eta_mean', flags='g')
    mean_eta_basin = int(round(float(eta_basin['mean'])))




    #ET Histogram
    ETa = raster2numpy('eta_mean', mapset='job{}'.format(jobid))
    # r.mask(raster="LC_studyarea", maskcats='2 thru 7')
    ETa = np.ma.masked_where(ETa == -2147483648, ETa)
    eta_hist_plt = os.path.join(newdir, "eta_crop_hist.png")
    eta_mean_values = ETa.compressed()
    print("eta_mean_values",eta_mean_values)


    plt.figure(figsize=(8, 4))
    plt.hist(eta_mean_values, bins=500, histtype='step', linewidth=1.5)
    plt.title('Histogram of Seasonal ETa', fontsize=12)
    plt.xlabel('ETa Values (mm)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.savefig(eta_hist_plt, bbox_inches='tight',pad_inches = 0.1, dpi=100)


    raster_data = ETa.data  # Get the underlying data array
    valid_et_data = raster_data[~np.isnan(raster_data)]
    eta_hist_plt2 = os.path.join(newdir, "eta_crop_hist2.png")
    et_min, et_max = np.floor(valid_et_data.min()), np.ceil(valid_et_data.max())  # Round to nearest integers
    et_min = round(et_min/100 , 0)*100
    et_max = round(et_max/100 , 0)*100
    print("et_min",et_min)
    print("et_max",et_max)

    et_bins = np.linspace(et_min, et_max, 11).astype(int)
    print("et_bins",et_bins)

    areas_per_bin = []
    for i in range(len(et_bins) - 1):
        # Count pixels in the current bin range
        pixels_in_bin = np.logical_and(valid_et_data >= et_bins[i], valid_et_data < et_bins[i + 1])
        # Calculate the area for this bin (assuming pixel size is in square km)
        area = pixels_in_bin.sum() * 0.09    # Adjust pixel_area if known
        areas_per_bin.append(area)
    

    plt.figure(figsize=(8, 4))
    plt.bar(range(10), areas_per_bin, width=0.9, color='blue', alpha=0.7, 
            tick_label=[f'{et_bins[i]}-{et_bins[i+1]}' for i in range(10)])
    plt.title('Histogram of Seasonal ETa', fontsize=12)
    plt.xlabel('ETa values (mm)')
    plt.ylabel('Area (sq km)')
    plt.savefig(eta_hist_plt2, bbox_inches='tight',pad_inches = 0.1, dpi=100)

    histcsv_filepath = os.path.join(newdir, "eta_histogram_data.csv")
    histogram_data = pd.DataFrame({
        'ETa Range': [f'{et_bins[i]}-{et_bins[i+1]}' for i in range(10)],
        'Area (sq km)': areas_per_bin
    })
    histogram_data.to_csv(histcsv_filepath, index=False)


    r.mask(flags="r")

    progress += 10
    self.update_state(state='PROGRESS', meta={'current': progress, 'total': total_steps})

    

    ### FIGURE 7  ETblue plot ###
    etbplt = os.path.join(newdir, "etb.png")
    etgplt = os.path.join(newdir, "etg.png")
    ETb=raster2numpy('etb_mean', mapset='job{}'.format(jobid))
    ETb = np.ma.masked_where(ETb == -2147483648, ETb)
    ETg=raster2numpy('etg_mean', mapset='job{}'.format(jobid))
    ETg = np.ma.masked_where(ETg == -2147483648, ETg)
    etbtif = os.path.join(newdir, "ETb.tif")
    r.out_gdal(input='etb_mean', output=etbtif)
    etgtif = os.path.join(newdir, "ETg.tif")
    r.out_gdal(input='etg_mean', output=etgtif)
    et_max = np.nanpercentile(np.maximum(ETb, ETg), 99)
    fig, ax = plt.subplots(figsize = (12,8))
    plt.imshow(ETb, cmap='jet_r', vmin=0, vmax=et_max, extent=spatial_extent, interpolation='none', resample=False)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
    x, y, arrow_length = 1.1, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.colorbar(shrink=0.50, label='ETblue [mm/year]')
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
    plt.title('Seasonal ET blue ', fontsize=12)
    plt.savefig(etbplt, bbox_inches='tight',pad_inches = 0, dpi=100)
    
    etb_basin = grass.parse_command('r.univar', map=f'etb_mean', flags='g')
    mean_etb_basin = int(round(float(etb_basin['mean'])))
    
    ### FIGURE 8  ETgreen plot ###
    fig, ax = plt.subplots(figsize = (12,8))
    plt.imshow(ETg, cmap='jet_r', vmin=0, vmax=et_max, extent=spatial_extent, interpolation='none', resample=False)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
    x, y, arrow_length = 1.1, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.colorbar(shrink=0.50, label='ET green [mm/year]')
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
    plt.title('Seasonal ET green', fontsize=12)
    plt.savefig(etgplt, bbox_inches='tight',pad_inches = 0, dpi=100)
    
    etg_basin = grass.parse_command('r.univar', map=f'etg_mean', flags='g')
    mean_etg_basin = int(round(float(etg_basin['mean'])))


    ## FIGURE xx - ETr plot ###
    etrplt = os.path.join(newdir, "etr.png")
    etrtif = os.path.join(newdir, "ETr.tif")
    r.out_gdal(input='etr_mean', output=etrtif)
    #print(etmaps)
    ETr=raster2numpy('etr_mean', mapset='job{}'.format(jobid))
    ETr = np.ma.masked_where(ETr == -2147483648, ETr)
    fig, ax = plt.subplots(figsize = (12,8))
    plt.imshow(ETr, cmap='jet_r', vmin=np.nanmin(ETr), vmax=np.nanmax(ETr),extent=spatial_extent, interpolation='none', resample=False)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
    x, y, arrow_length = 1.1, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.colorbar(shrink=0.50, label='ETr [mm/year]')
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
    plt.title('Seasonal ETr ', fontsize=12)
    plt.savefig(etrplt, bbox_inches='tight',pad_inches = 0, dpi=100)
    
    etr_basin = grass.parse_command('r.univar', map=f'etr_mean', flags='g')
    mean_etr_basin = int(round(float(etr_basin['mean'])))

    ## FIGURE 4 PCP plot###
    pcpplt = os.path.join(newdir, "pcp.png")
    pcptif = os.path.join(newdir, "PCP.tif")
    r.out_gdal(input='pcp_mean', output=pcptif)
    #print(etmaps)
    PCP=raster2numpy("pcp_mean", mapset='job{}'.format(jobid))
    PCP = np.ma.masked_where(PCP == -2147483648, PCP)
    fig, ax = plt.subplots(figsize = (12,8))
    plt.imshow(PCP, cmap='Blues', vmin=np.nanmin(PCP), vmax=np.nanmax(PCP),extent=spatial_extent, interpolation='none', resample=False)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
    x, y, arrow_length = 1.1, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.colorbar(shrink=0.50, label='Precipitation [mm/year]')
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
    plt.title('Seasonal Precipitation', fontsize=12)
    plt.savefig(pcpplt, bbox_inches='tight',pad_inches = 0, dpi=100)
    
    pcp_basin = grass.parse_command('r.univar', map=f'pcp_mean', flags='g')
    mean_pcp_basin = "%.0f" % round(float(pcp_basin['mean']), 1)
    
    ## FIGURE 5 PCP-ETa plot ###
    pminet = os.path.join(newdir, "pminuset.png")
    grass.mapcalc('{r} = {a} - {b}'.format(r=f'pminuset', a=f'pcp_mean', b=f'eta_mean'))
    #print(etmaps)
    pminusettif = os.path.join(newdir, "PminusET.tif")
    r.out_gdal(input='pminuset', output=pminusettif)
    pminuset=raster2numpy("pminuset", mapset='job{}'.format(jobid))
    pminuset = np.ma.masked_where(pminuset == -2147483648, pminuset)
    fig, ax = plt.subplots(figsize = (12,8))
    print(np.nanmin(pminuset))
    print(np.nanmax(pminuset))
    
    if np.nanmax(pminuset) > 0 and np.nanmin(pminuset) < 0:
           print('first condition for divnorm')
           divnorm=colors.TwoSlopeNorm(vmin=np.nanmin(pminuset), vcenter=0, vmax=np.nanmax(pminuset))
    else:
           print('second condition for divnorm')
           divnorm=colors.TwoSlopeNorm(vmin=np.nanmin(pminuset), vcenter=np.nanmedian(pminuset), vmax=np.nanmax(pminuset))   
    plt.imshow(pminuset, cmap='RdYlBu', extent=spatial_extent, norm=divnorm, interpolation='none', resample=False)
    #plt.imshow(pminuset, cmap='RdYlBu', vmin=-1000, vmax=np.nanmax(pminuset),extent=spatial_extent, interpolation='none', resample=False)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
    x, y, arrow_length = 1.1, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.colorbar(shrink=0.50, label='PCP - ETa [mm/year]')
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
    plt.title('PCP - ETa', fontsize=12)
    plt.savefig(pminet, bbox_inches='tight',pad_inches = 0, dpi=100)
    
    r.mask(raster="LC_studyarea", maskcats='2 thru 7')

    
    ## Saving table with Bio, ET and WP annual

    mapsdmp = ["wapor3_npp_" + s for s in years_str]
    dmp=[]
    print("mapsdmp",mapsdmp)
    for i in mapsdmp:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = round(float(stats['mean']) * 22.222 * 0.001,0)
            dmp.append(mean)
    print("dmp")
    print(dmp)
    etag=[]

    etmaps = ["wapor3_eta_" + s for s in years_str]
    for i in etmaps:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            #mean = int(round(float(stats['mean'])))
            mean = round(float(stats['mean']),0)
            etag.append(mean)
    print("etag")
    print(etag)
    WPbAnnual = [round(float(a / (b * 10)), 2) for a, b in zip(dmp, etag)]
    df_wpb = pd.DataFrame({'TBP(Kg/ha)': dmp, 'ETa(mm/year)': etag, 'WPb(Kg/m3)': WPbAnnual}, index=years)
    df_wpb.loc['Average'] = round(df_wpb.mean(), 1)
    dfwpb = os.path.join(newdir, "wpbtable.csv")
    df_wpb.to_csv(dfwpb, index = True)
    
    ## FIGURE 6 DMP plot ###
    dmpfig = os.path.join(newdir, "dmp.png")
    dmptif = os.path.join(newdir, "DMP.tif")
    grass.mapcalc('{r} = {a} * 22.222 * 0.001'.format(r=f'dmp_mean', a=f'npp_mean'))
    r.out_gdal(input='dmp_mean', output=dmptif)
    dmp=raster2numpy("dmp_mean", mapset='job{}'.format(jobid))
    dmp = np.ma.masked_where(dmp == -2147483648, dmp)
    fig, ax = plt.subplots(figsize = (12,8))
    plt.imshow(dmp, cmap='BrBG', vmin=1000, vmax=np.nanmax(dmp),extent=spatial_extent, interpolation='none', resample=False)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
    x, y, arrow_length = 1.1, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.colorbar(shrink=0.50, label='DMP [Kg/ha]')
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
    plt.title('Total Biomass Production (TBP)', fontsize=12)
    plt.savefig(dmpfig, bbox_inches='tight',pad_inches = 0, dpi=100)    
    dmp_basin = grass.parse_command('r.univar', map=f'dmp_mean', flags='g')
    #mean_dmp_basin = round(float(dmp_basin['mean']), 0)    


    r.mask(raster="LC_studyarea", maskcats='2 thru 7')
    ## RWD map ###
#     eta_irri = grass.parse_command('r.univar', map=f'eta_mean', flags='ge', percentile='98')
#     thrwd = round(float(eta_irri['percentile_98']), 0)
#     grass.mapcalc('{r} = 1 - ({a} / {b})'.format(r=f'rwd1', a=f'eta_mean', b=thrwd))
#     grass.mapcalc('{r} = if({a} < 0, null(), {a})'.format(r=f'rwd', a=f'rwd1'))
    #mean_eta_irri = round(float(eta_irri['mean']), 0)
    ## WPdmp map ##
    grass.mapcalc('{r} = {a} / ({b} * 10)'.format(r=f'WPdmp', b=f'eta_mean', a='dmp_mean'))


    r.mask(flags="r")
    g.region(vector=vectname, res=0.003)
    
    ##Plotting Ta and Ea
    r.mask(vector=vectname)
    eatif = os.path.join(newdir, "Ea.tif")
    r.out_gdal(input='ea_mean', output=eatif)
    tatif = os.path.join(newdir, "Ta.tif")
    r.out_gdal(input='ta_mean', output=tatif)
    r.mask(raster="LC_studyarea", maskcats='2 thru 7')
    grass.mapcalc('{r} = {a}'.format(r=f'E_mean_crop', a=f'ea_mean'))
    grass.mapcalc('{r} = {a}'.format(r=f'T_mean_crop', a=f'ta_mean'))
    Ea=raster2numpy('E_mean_crop', mapset='job{}'.format(jobid))
    Ea = np.ma.masked_where(Ea == -2147483648, Ea)
    Ta=raster2numpy('T_mean_crop', mapset='job{}'.format(jobid))
    Ta = np.ma.masked_where(Ta == -2147483648, Ta)
    e_t_max = np.nanpercentile(np.maximum(Ea, Ta), 99)
    taplt = os.path.join(newdir, "ta.png")
    eaplt = os.path.join(newdir, "ea.png")
    #print(etmaps)

    ## FIGURE XX - Ea plot ###
    fig, ax = plt.subplots(figsize = (12,8))
    plt.imshow(Ea, cmap='jet_r', vmin=0, vmax=e_t_max,extent=spatial_extent, interpolation='none', resample=False)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
    x, y, arrow_length = 1.1, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.colorbar(shrink=0.50, label='Ea [mm/year]')
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
    plt.title('Seasonal Ea ', fontsize=12)
    plt.savefig(eaplt, bbox_inches='tight',pad_inches = 0, dpi=100)

    ## FIGURE XX - Ta plot ###
    #print(etmaps)
    fig, ax = plt.subplots(figsize = (12,8))
    plt.imshow(Ta, cmap='jet_r', vmin=0, vmax=e_t_max,extent=spatial_extent, interpolation='none', resample=False)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k')
    x, y, arrow_length = 1.1, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.colorbar(shrink=0.50, label='Ta [mm/year]')
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
    plt.title('Seasonal Ta ', fontsize=12)
    plt.savefig(taplt, bbox_inches='tight',pad_inches = 0, dpi=100)
    lc_ea_stats = grass.parse_command('r.univar', map=f'ea_mean', zones=f'LC_studyarea', flags='gt')
    lc_ta_stats = grass.parse_command('r.univar', map=f'ta_mean', zones=f'LC_studyarea', flags='gt')
    
    ### FIGURE 9 Wpdmp plot ###
    wpplt = os.path.join(newdir, "wpdmp.png")
    wptif = os.path.join(newdir, "WPDMP.tif")
    r.out_gdal(input='WPdmp', output=wptif)    
    WPdmp=raster2numpy('WPdmp', mapset='job{}'.format(jobid))
    WPdmp = np.ma.masked_where(WPdmp == -2147483648, WPdmp)
    fig, ax = plt.subplots(figsize = (12,8))
    plt.imshow(WPdmp, cmap='RdYlGn', vmin=0, vmax=np.nanpercentile(WPdmp, 95),extent=spatial_extent, interpolation='none', resample=False)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
    x, y, arrow_length = 1.1, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.colorbar(shrink=0.50, label=' Water Productivity [Kg/m3]')
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
    plt.title('Seasonal Water Productivity', fontsize=12)
    plt.savefig(wpplt, bbox_inches='tight',pad_inches = 0, dpi=100)
    
    ### FIGURE 10 RWD plot ###
#     rwdplt = os.path.join(newdir, "rwd.png")
#     rwdtif = os.path.join(newdir, "WDI.tif")
#     r.out_gdal(input='rwd', output=rwdtif)
#     rwd=raster2numpy('rwd', mapset='job{}'.format(jobid))
#     rwd = np.ma.masked_where(rwd == -2147483648, rwd)
#     fig, ax = plt.subplots(figsize = (12,8))
#     plt.imshow(rwd, cmap='RdBu_r', vmin=0, vmax=np.nanpercentile(rwd, 99),extent=spatial_extent, interpolation='none', resample=False)
#     scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
#     fig.gca().add_artist(scalebar)
#     df.boundary.plot(ax=ax, facecolor='none', edgecolor='k');
#     x, y, arrow_length = 1.1, 0.1, 0.1
#     ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
#             arrowprops=dict(facecolor='black', width=5, headwidth=15),
#             ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
#     #ax.legend(bbox_to_anchor=(0.17,0.2))
#     plt.colorbar(shrink=0.50, label='WDI [%]')
#     plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
#     plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
#     plt.title('Water Deficit Index', fontsize=12)
#     plt.savefig(rwdplt, bbox_inches='tight',pad_inches = 0, dpi=100)
    
    
    
    r.mask(flags="r")
    r.mask(vector=vectname)
    ### Bar plots - land use versus ETa, PCP ###
    
    lc_eta_stats = grass.parse_command('r.univar', map=f'eta_mean', zones=f'LC_studyarea', flags='gt')
    lc_etb_stats = grass.parse_command('r.univar', map=f'etb_mean', zones=f'LC_studyarea', flags='gt')
    lc_etg_stats = grass.parse_command('r.univar', map=f'etg_mean', zones=f'LC_studyarea', flags='gt')
    lc_etr_stats = grass.parse_command('r.univar', map=f'etr_mean', zones=f'LC_studyarea', flags='gt')
    lc_pcp_stats = grass.parse_command('r.univar', map=f'pcp_mean', zones=f'LC_studyarea', flags='gt')
    #lc_etg_stats = grass.parse_command('r.univar', map=f'ETg_mean', zones=f'LC_studyarea', flags='gt')
    #lc_etb_stats = grass.parse_command('r.univar', map=f'ETb_mean', zones=f'LC_studyarea', flags='gt')
    neta = list(lc_eta_stats.keys())
    netb = list(lc_etb_stats.keys())
    netg = list(lc_etg_stats.keys())
    netr = list(lc_etr_stats.keys())
    npcp = list(lc_pcp_stats.keys())
#     nea = list(lc_ea_stats.keys())
#     nta = list(lc_ta_stats.keys())
    #netg = list(lc_etg_stats.keys())
    #netb = list(lc_etb_stats.keys())
    #  d=["%.0f" % round(float(item), 0) for item in a]
    yeta = [item for items in neta for item in items.split("|")]
    lc_eta_str = yeta[15::14]
    lc_eta_mean = [round(float(item),0) for item in yeta[21::14]]

    yetb = [item for items in netb for item in items.split("|")]
    lc_etb_mean = [round(float(item),0) for item in yetb[21::14]]

    yetg = [item for items in netg for item in items.split("|")]
    lc_etg_mean = [round(float(item),0) for item in yetg[21::14]]

    yetr = [item for items in netr for item in items.split("|")]
    lc_etr_mean = [round(float(item),0) for item in yetr[21::14]]
    ypcp = [item for items in npcp for item in items.split("|")]



    progress += 10
    self.update_state(state='PROGRESS', meta={'current': progress, 'total': total_steps})

    






    #lc_pcp_mean=["%.0f" % round(float(item), 0) for item in ypcp[21::14]]
    lc_pcp_mean = [round(float(item), 1) for item in ypcp[21::14]]
    lc_pcp_mean_str = ["%.0f" % round(float(item), 1) for item in lc_pcp_mean]
    print(f"PCP mean are {lc_pcp_mean}")
    #yea = [item for items in nea for item in items.split("|")]
    #lc_ea_mean = [int(round(float(item))) for item in yea[21::14]]
    #lc_ea_mean = yea[21::14]
    print(f"LC names are {lc_eta_str}")
    #print(f"Annual Ea LC stats {lc_ea_mean}")
    #yta = [item for items in nta for item in items.split("|")]
    #lc_ta_mean = [int(round(float(item))) for item in yta[21::14]]
    #lc_ta_mean = yta[21::14]
    #print(f"Annual Ea LC stats {lc_ta_mean}")
    #yetg = [item for items in netg for item in items.split("|")]
    #lc_etg_mean = [int(round(float(item))) for item in yetg[21::14]]
    #yetb = [item for items in netb for item in items.split("|")]
    #lc_etb_mean = [int(round(float(item))) for item in yetb[21::14]]

    lcbar = os.path.join(newdir, "lcbar.png")
    #x = np.arange(len(lc_eta_str))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    df1 = pd.DataFrame({'ETa': lc_eta_mean, 'PCP': lc_pcp_mean, 'Area': LC_area_sqkm}, index=lc_eta_str)
    df2 = df1.sort_values(by = ['Area'], ascending=False)
    ax = df2.plot.barh(y = ['ETa', 'PCP'], color=['seagreen', 'dodgerblue'])
    ax.invert_yaxis()
    ax.set_title('Seasonal ETa/PCP per Landcover')
    ax.set_xlabel('mm/year')
    plt.savefig(lcbar, bbox_inches='tight',pad_inches = 0.1, dpi=100)

    ### Bar plot - multiple years eta stats in one bar graph
    if et == 'ssebop' or et == 'enset' or et == 'wapor2':
            eta_2010 = grass.parse_command('r.univar', map=f'{et}_eta_y2010', zones=f'LC_studyarea', flags='gt')
            eta_2011 = grass.parse_command('r.univar', map=f'{et}_eta_y2011', zones=f'LC_studyarea', flags='gt')
            eta_2012 = grass.parse_command('r.univar', map=f'{et}_eta_y2012', zones=f'LC_studyarea', flags='gt')
            eta_2013 = grass.parse_command('r.univar', map=f'{et}_eta_y2013', zones=f'LC_studyarea', flags='gt')
            eta_2014 = grass.parse_command('r.univar', map=f'{et}_eta_y2014', zones=f'LC_studyarea', flags='gt')
            eta_2015 = grass.parse_command('r.univar', map=f'{et}_eta_y2015', zones=f'LC_studyarea', flags='gt')
            eta_2016 = grass.parse_command('r.univar', map=f'{et}_eta_y2016', zones=f'LC_studyarea', flags='gt')
            eta_2017 = grass.parse_command('r.univar', map=f'{et}_eta_y2017', zones=f'LC_studyarea', flags='gt')
            eta_2018 = grass.parse_command('r.univar', map=f'{et}_eta_y2018', zones=f'LC_studyarea', flags='gt')
            eta_2019 = grass.parse_command('r.univar', map=f'{et}_eta_y2019', zones=f'LC_studyarea', flags='gt')
            eta_2020 = grass.parse_command('r.univar', map=f'{et}_eta_y2020', zones=f'LC_studyarea', flags='gt')
            eta_2021 = grass.parse_command('r.univar', map=f'{et}_eta_y2021', zones=f'LC_studyarea', flags='gt')
            eta_2022 = grass.parse_command('r.univar', map=f'{et}_eta_y2022', zones=f'LC_studyarea', flags='gt')
            eta_2023 = grass.parse_command('r.univar', map=f'{et}_eta_y2023', zones=f'LC_studyarea', flags='gt')
            neta2010 = list(eta_2010.keys())
            neta2011 = list(eta_2011.keys())
            neta2012 = list(eta_2012.keys())
            neta2013 = list(eta_2013.keys())
            neta2014 = list(eta_2014.keys())
            neta2015 = list(eta_2015.keys())
            neta2016 = list(eta_2016.keys())
            neta2017 = list(eta_2017.keys())
            neta2018 = list(eta_2018.keys())
            neta2019 = list(eta_2019.keys())
            neta2020 = list(eta_2020.keys())
            neta2021 = list(eta_2021.keys())
            neta2022 = list(eta_2022.keys())
            neta2023 = list(eta_2023.keys())
            yeta2010 = [item for items in neta2010 for item in items.split("|")]
            yeta2011 = [item for items in neta2011 for item in items.split("|")]
            yeta2012 = [item for items in neta2012 for item in items.split("|")]
            yeta2013 = [item for items in neta2013 for item in items.split("|")]
            yeta2014 = [item for items in neta2014 for item in items.split("|")]
            yeta2015 = [item for items in neta2015 for item in items.split("|")]
            yeta2016 = [item for items in neta2016 for item in items.split("|")]
            yeta2017 = [item for items in neta2017 for item in items.split("|")]
            yeta2018 = [item for items in neta2018 for item in items.split("|")]
            yeta2019 = [item for items in neta2019 for item in items.split("|")]
            yeta2020 = [item for items in neta2020 for item in items.split("|")]
            yeta2021 = [item for items in neta2021 for item in items.split("|")]
            yeta2022 = [item for items in neta2022 for item in items.split("|")]
            yeta2023 = [item for items in neta2023 for item in items.split("|")]
            lc_eta_str = yeta2020[15::14]
            lc_eta_2010 = [round(float(item),0) for item in yeta2010[21::14]]
            lc_eta_2011 = [round(float(item),0) for item in yeta2011[21::14]]
            lc_eta_2012 = [round(float(item),0) for item in yeta2012[21::14]]
            lc_eta_2013 = [round(float(item),0) for item in yeta2013[21::14]]
            lc_eta_2014 = [round(float(item),0) for item in yeta2014[21::14]]
            lc_eta_2015 = [round(float(item),0) for item in yeta2015[21::14]]
            lc_eta_2016 = [round(float(item),0) for item in yeta2016[21::14]]
            lc_eta_2017 = [round(float(item),0) for item in yeta2017[21::14]]
            lc_eta_2018 = [round(float(item),0) for item in yeta2018[21::14]]
            lc_eta_2019 = [round(float(item),0) for item in yeta2019[21::14]]
            lc_eta_2020 = [round(float(item),0) for item in yeta2020[21::14]]
            lc_eta_2021 = [round(float(item),0) for item in yeta2021[21::14]]
            lc_eta_2022 = [round(float(item),0) for item in yeta2022[21::14]]
            lc_eta_2023 = [round(float(item),0) for item in yeta2023[21::14]]
            lcbaret = os.path.join(newdir, "lcbaret.png")
            #x = np.arange(len(lc_eta_str))  # the label locations
            #width = 0.35  # the width of the bars
            fig, ax = plt.subplots(figsize=(40,10))
            df = pd.DataFrame({'2010': lc_eta_2010, '2011': lc_eta_2011, '2012': lc_eta_2012, '2013': lc_eta_2013, '2014': lc_eta_2014, '2015': lc_eta_2015, '2016': lc_eta_2016, '2017': lc_eta_2017, '2018': lc_eta_2018, '2019': lc_eta_2019, '2020': lc_eta_2020, '2021': lc_eta_2021, '2022': lc_eta_2022, '2023': lc_eta_2023}, index=lc_eta_str)
            if et == 'wapor2' or et == 'wapor3':
                    for col in df.columns:
                            df[col] = df[col] * 0.1
            else:
                    print('No need of scaling')
            ax = df.plot.barh()
            ax.invert_yaxis()
            #ax.set_title('Yearly ETa per Landcover', fontsize=14)
            #ax.set_xlabel('mm/year', fontsize=18)
            plt.title('Yearly ETa per Landcover', fontsize=12)
            plt.xlabel('mm/year', fontsize=12)
            plt.savefig(lcbaret, bbox_inches='tight',pad_inches = 0.1, dpi=100)

    if et == 'nrsc':
            eta_2016 = grass.parse_command('r.univar', map=f'{et}_eta_y2016', zones=f'LC_studyarea', flags='gt')
            eta_2017 = grass.parse_command('r.univar', map=f'{et}_eta_y2017', zones=f'LC_studyarea', flags='gt')
            eta_2018 = grass.parse_command('r.univar', map=f'{et}_eta_y2018', zones=f'LC_studyarea', flags='gt')
            eta_2019 = grass.parse_command('r.univar', map=f'{et}_eta_y2019', zones=f'LC_studyarea', flags='gt')
            eta_2020 = grass.parse_command('r.univar', map=f'{et}_eta_y2020', zones=f'LC_studyarea', flags='gt')
            eta_2021 = grass.parse_command('r.univar', map=f'{et}_eta_y2021', zones=f'LC_studyarea', flags='gt')
            eta_2022 = grass.parse_command('r.univar', map=f'{et}_eta_y2022', zones=f'LC_studyarea', flags='gt')
            eta_2023 = grass.parse_command('r.univar', map=f'{et}_eta_y2023', zones=f'LC_studyarea', flags='gt')
            neta2016 = list(eta_2016.keys())
            neta2017 = list(eta_2017.keys())
            neta2018 = list(eta_2018.keys())
            neta2019 = list(eta_2019.keys())
            neta2020 = list(eta_2020.keys())
            neta2021 = list(eta_2021.keys())
            neta2022 = list(eta_2022.keys())
            neta2023 = list(eta_2023.keys())
            yeta2016 = [item for items in neta2016 for item in items.split("|")]
            yeta2017 = [item for items in neta2017 for item in items.split("|")]
            yeta2018 = [item for items in neta2018 for item in items.split("|")]
            yeta2019 = [item for items in neta2019 for item in items.split("|")]
            yeta2020 = [item for items in neta2020 for item in items.split("|")]
            yeta2021 = [item for items in neta2021 for item in items.split("|")]
            yeta2022 = [item for items in neta2022 for item in items.split("|")]
            yeta2023 = [item for items in neta2023 for item in items.split("|")]
            lc_eta_str = yeta2020[15::14]
            lc_eta_2016 = [round(float(item),0) for item in yeta2016[21::14]]
            lc_eta_2017 = [round(float(item),0) for item in yeta2017[21::14]]
            lc_eta_2018 = [round(float(item),0) for item in yeta2018[21::14]]
            lc_eta_2019 = [round(float(item),0) for item in yeta2019[21::14]]
            lc_eta_2020 = [round(float(item),0) for item in yeta2020[21::14]]
            lc_eta_2021 = [round(float(item),0) for item in yeta2021[21::14]]
            lc_eta_2022 = [round(float(item),0) for item in yeta2022[21::14]]
            lc_eta_2023 = [round(float(item),0) for item in yeta2023[21::14]]
            lcbaret = os.path.join(newdir, "lcbaret.png")
            #x = np.arange(len(lc_eta_str))  # the label locations
            #width = 0.35  # the width of the bars
            fig, ax = plt.subplots(figsize=(40,10))
            df = pd.DataFrame({ '2016': lc_eta_2016, '2017': lc_eta_2017, '2018': lc_eta_2018, '2019': lc_eta_2019, '2020': lc_eta_2020, '2021': lc_eta_2021, '2022': lc_eta_2022, '2023': lc_eta_2023}, index=lc_eta_str)
            if et == 'wapor2' or et == 'wapor3':
                    for col in df.columns:
                            df[col] = df[col] * 0.1
            else:
                    print('No need of scaling')
            ax = df.plot.barh()
            ax.invert_yaxis()
            #ax.set_title('Yearly ETa per Landcover', fontsize=14)
            #ax.set_xlabel('mm/year', fontsize=18)
            plt.title('Yearly ETa per Landcover', fontsize=12)
            plt.xlabel('mm/year', fontsize=12)
            plt.savefig(lcbaret, bbox_inches='tight',pad_inches = 0.1, dpi=100)
    else:
            eta_2018 = grass.parse_command('r.univar', map=f'{et}_eta_y2018', zones=f'LC_studyarea', flags='gt')
            eta_2019 = grass.parse_command('r.univar', map=f'{et}_eta_y2019', zones=f'LC_studyarea', flags='gt')
            eta_2020 = grass.parse_command('r.univar', map=f'{et}_eta_y2020', zones=f'LC_studyarea', flags='gt')
            eta_2021 = grass.parse_command('r.univar', map=f'{et}_eta_y2021', zones=f'LC_studyarea', flags='gt')
            eta_2022 = grass.parse_command('r.univar', map=f'{et}_eta_y2022', zones=f'LC_studyarea', flags='gt')
            eta_2023 = grass.parse_command('r.univar', map=f'{et}_eta_y2023', zones=f'LC_studyarea', flags='gt')
            neta2018 = list(eta_2018.keys())
            neta2019 = list(eta_2019.keys())
            neta2020 = list(eta_2020.keys())
            neta2021 = list(eta_2021.keys())
            neta2022 = list(eta_2022.keys())
            neta2023 = list(eta_2023.keys())
            yeta2018 = [item for items in neta2018 for item in items.split("|")]
            yeta2019 = [item for items in neta2019 for item in items.split("|")]
            yeta2020 = [item for items in neta2020 for item in items.split("|")]
            yeta2021 = [item for items in neta2021 for item in items.split("|")]
            yeta2022 = [item for items in neta2022 for item in items.split("|")]
            yeta2023 = [item for items in neta2023 for item in items.split("|")]
            lc_eta_str = yeta2020[15::14]
            lc_eta_2018 = [round(float(item),0) for item in yeta2018[21::14]]
            lc_eta_2019 = [round(float(item),0) for item in yeta2019[21::14]]
            lc_eta_2020 = [round(float(item),0) for item in yeta2020[21::14]]
            lc_eta_2021 = [round(float(item),0) for item in yeta2021[21::14]]
            lc_eta_2022 = [round(float(item),0) for item in yeta2022[21::14]]
            lc_eta_2023 = [round(float(item),0) for item in yeta2023[21::14]]
            lcbaret = os.path.join(newdir, "lcbaret.png")
            #x = np.arange(len(lc_eta_str))  # the label locations
            #width = 0.35  # the width of the bars
            fig, ax = plt.subplots(figsize=(40,10))
            df = pd.DataFrame({'2018': lc_eta_2018, '2019': lc_eta_2019, '2020': lc_eta_2020, '2021': lc_eta_2021, '2022': lc_eta_2022, '2023': lc_eta_2023}, index=lc_eta_str)
            if et == 'wapor2' or et == 'wapor3':
                    for col in df.columns:
                            df[col] = df[col] * 0.1
            else:
                    print('No need of scaling')
            ax = df.plot.barh()
            ax.invert_yaxis()
            #ax.set_title('Yearly ETa per Landcover', fontsize=14)
            #ax.set_xlabel('mm/year', fontsize=18)
            plt.title('Yearly ETa per Landcover', fontsize=12)
            plt.xlabel('mm/year', fontsize=12)
            plt.savefig(lcbaret, bbox_inches='tight',pad_inches = 0.1, dpi=100)
    



    ## Table 2 Saving to csv's
    ## below eta and pcp in km3 vol
    # round(float(bbox['w']), 2)
    eta_vol = ["%.2f" % round(float(a / 1000 * b), 2) for a, b in zip(lc_eta_mean, LC_area_sqkm)]
    etb_vol = ["%.2f" % round(float(a / 1000 * b), 2) for a, b in zip(lc_etb_mean, LC_area_sqkm)]
    etg_vol = ["%.2f" % round(float(a / 1000 * b), 2) for a, b in zip(lc_etg_mean, LC_area_sqkm)]
#     etr_vol = ["%.2f" % round(float(a / 1000 * b), 2) for a, b in zip(lc_etr_mean, LC_area_sqkm)]
    pcp_vol = ["%.2f" % round(float(a / 1000 * b), 2) for a, b in zip(lc_pcp_mean, LC_area_sqkm)]
#     peta = [a - b for a, b in zip(lc_pcp_mean, lc_eta_mean)]
#     peta_str = ["%.1f" % round(float(item), 1) for item in peta]
#     peta_vol = ["%.2f" % round(float(a / 1000 * b), 2) for a, b in zip(peta, LC_area_sqkm)]
    #peta_perc = [int(round(a / b * 100)) for a, b in zip(peta, lc_pcp_mean)]
    df_bcm1 = pd.DataFrame({'Land cover type': lc_eta_str, 'Area(km\u00b2)': LC_area_sqkm, 'Area(%)': LC_perc_flt, 'P(mm)': lc_pcp_mean_str, 'ETa(mm)': lc_eta_mean,'ETb(mm)': lc_etb_mean,'ETg(mm)': lc_etg_mean,  }, index=lc_eta_str)
    df_bcm2 = df_bcm1.sort_values(by = ['Area(%)'], ascending=False)
    dfbcm2 = os.path.join(newdir, "Table3.csv")
    df_bcm2.to_csv(dfbcm2, index = False)
    print('Saving  Table 3')

#     df_bcm3 = pd.DataFrame({'Land cover type': lc_eta_str, 'Area(km\u00b2)': LC_area_sqkm, 'Area(%)': LC_perc_flt, 'P(km\u00b3)': pcp_vol, 'ETa(km\u00b3)': eta_vol,'ETb(km\u00b3)': etb_vol,'ETg(km\u00b3)': etg_vol, }, index=lc_eta_str)
    df_bcm3 = pd.DataFrame({'Land cover type': lc_eta_str, 'Area(km\u00b2)': LC_area_sqkm, 'Area(%)': LC_perc_flt, 'P(1000 m\u00b3)': pcp_vol, 'ETa(1000 m\u00b3)': eta_vol,'ETb(1000 m\u00b3)': etb_vol,'ETg(1000 m\u00b3)': etg_vol }, index=lc_eta_str)
    df_bcm4 = df_bcm3.sort_values(by = ['Area(%)'], ascending=False)
    dfbcm4 = os.path.join(newdir, "Table4.csv")
    df_bcm4.to_csv(dfbcm4, index = False)
    print('Saving  Table 4')

    # Table 5 with Ea Ta ETb ETg per LC
    #ea_vol = ["%.2f" % round(float(a / 1000000 * b), 2) for a, b in zip(lc_ea_mean, LC_area_sqkm)]
    #ta_vol = ["%.2f" % round(float(a / 1000000 * b), 2) for a, b in zip(lc_ta_mean, LC_area_sqkm)]
    #etg_vol = ["%.2f" % round(float(a / 1000000 * b), 2) for a, b in zip(lc_etg_mean, LC_area_sqkm)]
    #etb_vol = ["%.2f" % round(float(a / 1000000 * b), 2) for a, b in zip(lc_etb_mean, LC_area_sqkm)]
    #df_bcm5 = pd.DataFrame({'Land cover type': lc_eta_str, 'Area(km\u00b2)': LC_area_sqkm, 'Area(%)': LC_perc_flt, 'Ea(mm/year)': lc_ea_mean, 'Ta(mm/year)': lc_ta_mean}, index=lc_eta_str)
    #df_bcm5['Ea(mm/year)'].astype("Int32")
    #df_bcm5['Ta(mm/year)'].astype("Int32")
    #df_bcm5['Ea(mm/year)'] = np.floor(pd.to_numeric(df_bcm5['Ea(mm/year)'], downcast='float', errors='coerce').astype('Int64'))
    #df_bcm5['Ta(mm/year)'] = np.floor(pd.to_numeric(df_bcm5['Ta(mm/year)'], downcast='float', errors='coerce').astype('Int64'))
    #df_bcm6 = df_bcm5.sort_values(by = ['Area(%)'], ascending=False)
    #dfbcm6 = os.path.join(newdir, "Table5.csv")
    #df_bcm6.to_csv(dfbcm6, index = False)
    #print('Saving  Table 5')
    
    LC0 = df_bcm4.iat[0,0]
    LC1 = df_bcm4.iat[1,0]
    LCA0 = df_bcm4.iat[0,1]
    LCA1 = df_bcm4.iat[1,1]
    ### COMPARISON bar charts ###
    #year = list(range(2009,2023))
    
#     maps_eta = [et + "_eta_y" + s for s in years_str]
#     eta=[]
#     for i in maps_eta:
#             stats = grass.parse_command('r.univar', map=i, flags='g')
#             mean = round(float(stats['mean']), 0)
#             eta.append(mean)
#     print('etamean:')
#     print(eta)


    progress += 10
    self.update_state(state='PROGRESS', meta={'current': progress, 'total': total_steps})

    maps_eta = [et + "_eta_y" + s for s in years_str]
    eta=[]
    for i in maps_eta:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = round(float(stats['mean']), 0)
            eta.append(mean)
    print('etamean:')

    
    if et == 'wapor2' or et == 'wapor3':
       eta = [x * 0.1 for x in eta]
    else:
       eta = eta
    print('etamean scaled:')
    print(eta)



    eta_s=[]
    for i in mapseta:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = round(float(stats['mean']), 0)
            eta_s.append(mean)
    print('etamean seasonal:')
    print(eta_s)



    
 

    ##ssebop_etpa_y2016
    maps_ssebopetr = ["ssebop_etpa_y" + s for s in years_str]
    ssebopetr=[]
    for i in maps_ssebopetr:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = round(float(stats['mean']), 0)
            ssebopetr.append(mean)
    print('ssebopetr:')
    print(ssebopetr)


    ssebopetr_s=[]
    for i in mapsetp:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = round(float(stats['mean']), 0)
            ssebopetr_s.append(mean)
    print('mean ssebopetr seasonal:')
    print(ssebopetr_s)

# avg seasonal ETb calculation
    
    etb_s=[]
    for i in mapsetb:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = round(float(stats['mean']), 0)
            etb_s.append(mean)
    print('mean ETb seasonal:')
    print(etb_s)

    # maps_wapor2 = ["wapor2_eta_y" + s for s in years_str]
    # wap2=[]
    # for i in maps_wapor2:
            # stats = grass.parse_command('r.univar', map=i, flags='g')
            # mean = round(float(stats['mean']), 0)
            # wap2.append(mean)
    # wapor2 = [x * 0.1 for x in wap2]
    # print('wapor2:')
    # print(wapor2)
    
    # maps_wapor3 = ["wapor3_eta_y" + s for s in years_str]
    # wap3=[]
    # for i in maps_wapor3:
            # stats = grass.parse_command('r.univar', map=i, flags='g')
            # mean = round(float(stats['mean']), 0)
            # wap3.append(mean)
    # wapor3 = [x * 0.1 for x in wap3]
    # print('wapor3:')
    # print(wapor3)

    # maps_ta = [f'Ta_{et}_annual_' + s for s in years_str]
    # ta1=[]
    # for i in maps_ta:
            # stats = grass.parse_command('r.univar', map=i, flags='g')
            # mean = int(round(float(stats['mean'])))
            # ta1.append(mean)

    # if et == 'wapor2' or et == 'wapor3':
        # ta = [x * 0.1 for x in ta1]
    # else:
        # ta = ta1
    # print('ta:')
    # print(ta)

    # maps_ea = [f'Ea_{et}_annual_' + s for s in years_str]
    # ea1=[]
    # for i in maps_ea:
            # stats = grass.parse_command('r.univar', map=i, flags='g')
            # mean = int(round(float(stats['mean'])))
            # ea1.append(mean)

    # if et == 'wapor2' or et == 'wapor3':
        # ea = [x * 0.1 for x in ea1]
    # else:
        # ea = ea1
    # print('ea:')
    # print(ea)
    
    #maps_modis=grass.list_grouped(type=['raster'], pattern="modis_eta_*")['data_annual']
    # maps_modis = ["modiseta_annual_" + s for s in years_str]
    # modis=[]
    # for i in maps_modis:
            # stats = grass.parse_command('r.univar', map=i, flags='g')
            # mean = int(round(float(stats['mean'])))
            # modis.append(mean)
    # print(modis)
    
    #maps_chirps=grass.list_grouped(type=['raster'], pattern="chirps_precip_*")['data_annual']

#     maps_chirps = ["pcpa_chirps_" + s for s in years_str]
#     chirps=[]
#     for i in maps_chirps:
#             stats = grass.parse_command('r.univar', map=i, flags='g')
#             mean = round(float(stats['mean']), 0)
#             chirps.append(mean)
#     print('chirps:')
#     print(chirps)

#     maps_ensindpcp = ["pcpa_ensind_" + s for s in years_str]
#     ensind=[]
#     for i in maps_ensindpcp:
#             stats = grass.parse_command('r.univar', map=i, flags='g')
#             mean = round(float(stats['mean']), 0)
#             ensind.append(mean)
#     print('ensind:')
#     print(ensind)
    
#     #maps_gpm=grass.list_grouped(type=['raster'], pattern="gpm_precip_*")['data_annual']
#     maps_gpm = ["pcpa_gpm_" + s for s in years_str]
#     gpm=[]
#     for i in maps_gpm:
#             stats = grass.parse_command('r.univar', map=i, flags='g')
#             mean = int(round(float(stats['mean'])))
#             gpm.append(mean)
#     print('gpm:')
#     print(gpm)
    
    #maps_persiann=grass.list_grouped(type=['raster'], pattern="persiann_precip_*")['data_annual']
#     maps_persiann = ["pcpa_persiann_" + s for s in years_str]
#     persiann=[]
#     for i in maps_persiann:
#             stats = grass.parse_command('r.univar', map=i, flags='g')
#             mean = int(round(float(stats['mean'])))
#             persiann.append(mean)
#     print('persiann:')
#     print(persiann)
    
#     maps_gsmap = ["pcpa_gsmap_" + s for s in years_str]
#     gsmap=[]
#     for i in maps_gsmap:
#             stats = grass.parse_command('r.univar', map=i, flags='g')
#             mean = int(round(float(stats['mean'])))
#             gsmap.append(mean)
#     print('gsmap:')
#     print(gsmap)
    
#     maps_era5 = ["pcpa_era5_" + s for s in years_str]
#     era5=[]
#     for i in maps_era5:
#             stats = grass.parse_command('r.univar', map=i, flags='g')
#             mean = int(round(float(stats['mean'])))
#             era5.append(mean)
#     print('era5:')
#     print(era5)
   


    maps_imd = ["pcpa_imd_" + s for s in years_str]
    imd=[]
    for i in maps_imd:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = int(round(float(stats['mean'])))
            imd.append(mean)
    print('imd_annual:')
    print(imd)


    imd_s=[]
    for i in mapspcp:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = int(round(float(stats['mean'])))
            imd_s.append(mean)
    print('imd_seasonal:')
    print(imd_s)




     ########IPA INDICATORS - MAPS & SUMMARY TABLE##############
    
    ## Actual Cropped Area ###
    statsdmp = grass.parse_command('r.univar', map=f'dmp_mean', flags='ge', percentile='25')
    thrwddmp = round(float(statsdmp['percentile_25']), 1)
    grass.mapcalc('{r} = if({a} < {b}, null(), {a})'.format(r=f'actcrop', a=f'dmp_mean', b=thrwddmp)) 
    grass.mapcalc('{r} = int(1)'.format(r=f'actcrop'))
    
    equity_kharif=[]
    adequacy_kharif=[]
    tbp_kharif=[]
    wp_kharif=[]
    actcrop_kharif=[]
    cropInten_kharif=[]

    equity_rabi=[]
    adequacy_rabi=[]
    tbp_rabi=[]
    wp_rabi=[]
    actcrop_rabi=[]
    cropInten_rabi=[]

    equity_zaid=[]
    adequacy_zaid=[]
    tbp_zaid=[]
    wp_zaid=[]
    actcrop_zaid=[]
    cropInten_zaid=[]

    equity_double=[]
    adequacy_double=[]
    tbp_double=[]
    wp_double=[]
    actcrop_double=[]
    cropInten_double=[]

    equity_fallow=[]
    adequacy_fallow=[]
    tbp_fallow=[]
    wp_fallow=[]
    actcrop_fallow=[]
    cropInten_fallow=[]
    
    equity_orchard=[]
    adequacy_orchard=[]
    tbp_orchard=[]
    wp_orchard=[]
    actcrop_orchard=[]
    cropInten_orchard=[]

    summary_csv_data={
          "Adequecy": [],
          "Equity": [],
          "TBP": [],
          "WP": [],
          "crop_area": [],
          "crop_intensity": [],
          
    }

    def calculate_average(values):
        return round(np.mean(values), 1) if len(values) > 0 else 0
    
    def avg_array_of_arrays(arrays):
        averages = [round(np.mean(array), 1) for array in arrays]  # Round to 1 decimal
        return averages

    filtered_in = [i for i, code in enumerate(LC_code) if 2 <= code <= 7]
    print("filtered_indices",filtered_in)
    filtered_LC_crop = [LC_name[i] for i in filtered_in]
    print("filtered_LC_crop",filtered_LC_crop)
    filtered_LC_area = [LC_area_sqkm[i] for i in filtered_in]


    if 2 in LC_code:
        r.mask(raster="LC_studyarea", maskcats='2')
        stats = grass.parse_command('r.univar', map=f'eta_mean', flags='ge', percentile='98')
        thrwd = round(float(stats['percentile_98']), 1)
        grass.mapcalc('{r} = {a} / {b}'.format(r=f'adeq_kharif', a=f'eta_mean', b=thrwd)) 

        fil_i = [i for i, code in enumerate(LC_code) if code == 2]
        class_area = float(LC_area_sqkm[fil_i[0]])
        print("class 2 area",class_area)

        for i in mapseta:
                yr = i.split("_")[2]
                stats = grass.parse_command('r.univar', map=i, flags='ge', percentile='98')
                thrwd = round(float(stats['percentile_98']), 1)
                mean = round(float(stats['mean']), 0)
                ad = round((mean/thrwd)*100, 0)
                cv = 100-round(float(stats['coeff_var']), 1)
                sd = round(float(stats['stddev']), 1)
                equity_kharif.append(cv)
                adequacy_kharif.append(ad)
                npp = et + "_npp_" + yr
                statsnpp = grass.parse_command('r.univar', map=npp, flags='g')
                meantbp = round(float(statsnpp['mean'])*0.001*22.222, 0)
                wp = meantbp/(mean * 0.001*10000)        
                tbp_kharif.append(meantbp)
                wp_kharif.append(wp)
                statscrop = grass.parse_command('r.univar', map='actcrop', flags='g')
                countcrop = round(float(statsnpp['n']), 0)
                areacrop = (countcrop * 300 * 300) / 1e6 ## active cropland area in sq.km
                actcrop_kharif.append(areacrop)
                cropInten_kharif.append(round(areacrop/class_area ,1))

        summary_csv_data['Equity'].append(equity_kharif)
        summary_csv_data['Adequecy'].append(adequacy_kharif)
        summary_csv_data['TBP'].append(tbp_kharif)
        summary_csv_data['WP'].append(wp_kharif)
        summary_csv_data['crop_area'].append(actcrop_kharif)
        summary_csv_data['crop_intensity'].append(cropInten_kharif)


    if 3 in LC_code:
        r.mask(raster="LC_studyarea", maskcats='3')
        stats = grass.parse_command('r.univar', map=f'eta_mean', flags='ge', percentile='98')
        thrwd = round(float(stats['percentile_98']), 1)
        grass.mapcalc('{r} = {a} / {b}'.format(r=f'adeq_rabi', a=f'eta_mean', b=thrwd))   
        fil_i = [i for i, code in enumerate(LC_code) if code == 3]
        class_area = float(LC_area_sqkm[fil_i[0]])
        print("class 3 area",class_area)
        for i in mapseta:
                stats = grass.parse_command('r.univar', map=i, flags='ge', percentile='98')
                thrwd = round(float(stats['percentile_98']), 1)
                mean = round(float(stats['mean']), 0)
                ad = round((mean/thrwd)*100, 0)
                cv = 100-round(float(stats['coeff_var']), 1)
                sd = round(float(stats['stddev']), 1)
                equity_rabi.append(cv)
                adequacy_rabi.append(ad)
                npp = et + "_npp_" + yr
                statsnpp = grass.parse_command('r.univar', map=npp, flags='g')
                meantbp = round(float(statsnpp['mean'])*0.001*22.222, 0)
                wp = meantbp/(mean * 0.001*10000)          
                tbp_rabi.append(meantbp)
                wp_rabi.append(wp)
                statscrop = grass.parse_command('r.univar', map='actcrop', flags='g')
                countcrop = round(float(statsnpp['n']), 0)
                areacrop = (countcrop * 300 * 300) / 1e6  ## active cropland area in sq.km
                actcrop_rabi.append(areacrop)
                cropInten_rabi.append(round(areacrop/class_area ,1))

        summary_csv_data['Equity'].append(equity_rabi)
        summary_csv_data['Adequecy'].append(adequacy_rabi)
        summary_csv_data['TBP'].append(tbp_rabi)
        summary_csv_data['WP'].append(wp_rabi)
        summary_csv_data['crop_area'].append(actcrop_rabi)
        summary_csv_data['crop_intensity'].append(cropInten_rabi)


    if 4 in LC_code:
        r.mask(raster="LC_studyarea", maskcats='4')
        stats = grass.parse_command('r.univar', map=f'eta_mean', flags='ge', percentile='98')
        print(stats)
        thrwd = round(float(stats['percentile_98']), 1)
        grass.mapcalc('{r} = {a} / {b}'.format(r=f'adeq_zaid', a=f'eta_mean', b=thrwd))   
        fil_i = [i for i, code in enumerate(LC_code) if code == 4]
        class_area = float(LC_area_sqkm[fil_i[0]])
        print("class 4 area",class_area)
        for i in mapseta:
                stats = grass.parse_command('r.univar', map=i, flags='ge', percentile='98')
                thrwd = round(float(stats['percentile_98']), 1)
                mean = round(float(stats['mean']), 0)
                ad = round((mean/thrwd)*100, 0)
                cv = 100-round(float(stats['coeff_var']), 1)
                sd = round(float(stats['stddev']), 1)
                equity_zaid.append(cv)
                adequacy_zaid.append(ad)
                npp = et + "_npp_" + yr
                statsnpp = grass.parse_command('r.univar', map=npp, flags='g')
                meantbp = round(float(statsnpp['mean'])*0.001*22.222, 0)
                wp = meantbp/(mean * 0.001*10000)         
                tbp_zaid.append(meantbp)
                wp_zaid.append(wp)
                statscrop = grass.parse_command('r.univar', map='actcrop', flags='g')
                countcrop = round(float(statsnpp['n']), 0)
                areacrop = (countcrop * 300 * 300) / 1e6  ## active cropland area in sq.km
                actcrop_zaid.append(areacrop)
                cropInten_zaid.append(round(areacrop/class_area ,1))

        summary_csv_data['Equity'].append(equity_zaid)
        summary_csv_data['Adequecy'].append(adequacy_zaid)
        summary_csv_data['TBP'].append(tbp_zaid)
        summary_csv_data['WP'].append(wp_zaid)
        summary_csv_data['crop_area'].append(actcrop_zaid)
        summary_csv_data['crop_intensity'].append(cropInten_zaid)


    if 5 in LC_code:
        r.mask(raster="LC_studyarea", maskcats='5')
        stats = grass.parse_command('r.univar', map=f'eta_mean', flags='ge', percentile='98')
        thrwd = round(float(stats['percentile_98']), 1)
        grass.mapcalc('{r} = {a} / {b}'.format(r=f'adeq_double', a=f'eta_mean', b=thrwd))   
        fil_i = [i for i, code in enumerate(LC_code) if code == 5]
        class_area = float(LC_area_sqkm[fil_i[0]])
        print("class 5 area",class_area)
        for i in mapseta:
                stats = grass.parse_command('r.univar', map=i, flags='ge', percentile='98')
                thrwd = round(float(stats['percentile_98']), 1)
                mean = round(float(stats['mean']), 0)
                ad = round((mean/thrwd)*100, 0)
                cv = 100-round(float(stats['coeff_var']), 1)
                sd = round(float(stats['stddev']), 1)
                equity_double.append(cv)
                adequacy_double.append(ad)
                npp = et + "_npp_" + yr
                statsnpp = grass.parse_command('r.univar', map=npp, flags='g')
                meantbp = round(float(statsnpp['mean'])*0.001*22.222, 0)
                wp = meantbp/(mean * 0.001*10000)        
                tbp_double.append(meantbp)
                wp_double.append(wp)
                statscrop = grass.parse_command('r.univar', map='actcrop', flags='g')
                countcrop = round(float(statsnpp['n']), 0)
                areacrop = (countcrop * 300 * 300) / 1e6  ## active cropland area in sq.km
                actcrop_double.append(areacrop)
                cropInten_double.append(round(areacrop/class_area ,1))

        summary_csv_data['Equity'].append(equity_double)
        summary_csv_data['Adequecy'].append(adequacy_double)
        summary_csv_data['TBP'].append(tbp_double)
        summary_csv_data['WP'].append(wp_double)
        summary_csv_data['crop_area'].append(actcrop_double)
        summary_csv_data['crop_intensity'].append(cropInten_double)



    if 6 in LC_code:
        r.mask(raster="LC_studyarea", maskcats='6')
        fil_i = [i for i, code in enumerate(LC_code) if code == 6]
        class_area = float(LC_area_sqkm[fil_i[0]])
        print("class 6 area",class_area)
        for i in mapseta:
                stats = grass.parse_command('r.univar', map=i, flags='ge', percentile='98')
                thrwd = round(float(stats['percentile_98']), 1)
                mean = round(float(stats['mean']), 0)
                ad = round((mean/thrwd)*100, 0)
                cv = 100-round(float(stats['coeff_var']), 1)
                sd = round(float(stats['stddev']), 1)
                equity_fallow.append(cv)
                adequacy_fallow.append(ad)
                npp = et + "_npp_" + yr
                statsnpp = grass.parse_command('r.univar', map=npp, flags='g')
                meantbp = round(float(statsnpp['mean'])*0.001*22.222, 0)
                wp = meantbp/(mean * 0.001*10000)        
                tbp_fallow.append(meantbp)
                wp_fallow.append(wp)
                statscrop = grass.parse_command('r.univar', map='actcrop', flags='g')
                countcrop = round(float(statsnpp['n']), 0)
                areacrop = (countcrop * 300 * 300) / 1e6  ## active cropland area in sq.km
                actcrop_fallow.append(areacrop)
                cropInten_fallow.append(round(areacrop/class_area ,1))

        summary_csv_data['Equity'].append(equity_fallow)
        summary_csv_data['Adequecy'].append(adequacy_fallow)
        summary_csv_data['TBP'].append(tbp_fallow)
        summary_csv_data['WP'].append(wp_fallow)
        summary_csv_data['crop_area'].append(actcrop_fallow)
        summary_csv_data['crop_intensity'].append(cropInten_fallow)


    if 7 in LC_code:
        r.mask(raster="LC_studyarea", maskcats='7')
        stats = grass.parse_command('r.univar', map=f'eta_mean', flags='ge', percentile='98')
        thrwd = round(float(stats['percentile_98']), 1)
        grass.mapcalc('{r} = {a} / {b}'.format(r=f'adeq_orchard', a=f'eta_mean', b=thrwd))   
        fil_i = [i for i, code in enumerate(LC_code) if code == 7]
        class_area = float(LC_area_sqkm[fil_i[0]])
        print("class 7 area",class_area)
        for i in mapseta:
                stats = grass.parse_command('r.univar', map=i, flags='ge', percentile='98')
                thrwd = round(float(stats['percentile_98']), 1)
                mean = round(float(stats['mean']), 0)
                ad = round((mean/thrwd)*100, 0)
                cv = 100-round(float(stats['coeff_var']), 1)
                sd = round(float(stats['stddev']), 1)
                equity_orchard.append(cv)
                adequacy_orchard.append(ad)
                npp = et + "_npp_" + yr
                statsnpp = grass.parse_command('r.univar', map=npp, flags='g')
                meantbp = round(float(statsnpp['mean'])*0.001*22.222, 0)
                wp = meantbp/(mean * 0.001*10000)       
                tbp_orchard.append(meantbp)
                wp_orchard.append(wp)
                statscrop = grass.parse_command('r.univar', map='actcrop', flags='g')
                countcrop = round(float(statsnpp['n']), 0)
                areacrop = (countcrop * 300 * 300) / 1e6
                actcrop_orchard.append(areacrop)
                cropInten_orchard.append(round(areacrop/class_area ,1))

        summary_csv_data['Equity'].append(equity_orchard)
        summary_csv_data['Adequecy'].append(adequacy_orchard)
        summary_csv_data['TBP'].append(tbp_orchard)
        summary_csv_data['WP'].append(wp_orchard)
        summary_csv_data['crop_area'].append(actcrop_orchard)
        summary_csv_data['crop_intensity'].append(cropInten_orchard)


    # Annual summary table
    rows = []
    for year_idx, year in enumerate(years):
        for crop_idx, crop in enumerate(filtered_LC_crop):
            row = [
                year,
                crop,
                filtered_LC_area[crop_idx], 
                summary_csv_data['Adequecy'][crop_idx][year_idx],
                summary_csv_data['Equity'][crop_idx][year_idx],
                summary_csv_data['TBP'][crop_idx][year_idx],
                summary_csv_data['WP'][crop_idx][year_idx],
                summary_csv_data['crop_area'][crop_idx][year_idx],
                # summary_csv_data['crop_intensity'][crop_idx][year_idx],
            ]
            rows.append(row)

    # Convert to DataFrame
    # df_summary_stats = pd.DataFrame(rows, columns=['Year', 'Crop Class','Area (km2)','Adequecy (%)', 'Equity (%)', 'Biomass land Productivity (kg/ha)', 'Biomass Water Productivity (kg/m3)', 'Actual cropped area (km2)','Cropping intensity'])
    df_summary_stats = pd.DataFrame(rows, columns=['Year', 'Crop Class','Area (km2)','Adequecy (%)', 'Equity (%)', 'Biomass land Productivity (kg/ha)', 'Biomass Water Productivity (kg/m3)', 'Actual cropped area (km2)'])
    
    dfbcm3 = os.path.join(newdir, "annual_summary_table.csv")
    df_summary_stats.to_csv(dfbcm3, index = False)
    print('Saving annual summary table')


    #     Average Summary table:
    df_bcm1 = pd.DataFrame({'Crop class': filtered_LC_crop, 'Area(km\u00b2)': filtered_LC_area, 'Equity(%)': avg_array_of_arrays(summary_csv_data['Equity']), 
                            'Adequcy(%)': avg_array_of_arrays(summary_csv_data['Adequecy']),'Biomass land Productivity (kg/ha)': avg_array_of_arrays(summary_csv_data['TBP']),
                            'Biomass Water Productivity (kg/m\u00b3)': avg_array_of_arrays(summary_csv_data['WP']), 'Actual cropped area (km\u00b2)':avg_array_of_arrays(summary_csv_data['crop_area']), 
                            # 'Cropping intensity':avg_array_of_arrays(summary_csv_data['crop_intensity'])
                            }, index=filtered_LC_crop)
    # Calculate the "Overall" row
    overall_row = [
        'Overall', 
        round(sum(df_bcm1['Area(km\u00b2)']),1),  
        round(df_bcm1['Equity(%)'].mean(),1), 
        round(df_bcm1['Adequcy(%)'].mean(),1), 
        round(df_bcm1['Biomass land Productivity (kg/ha)'].mean(),1), 
        round(df_bcm1['Biomass Water Productivity (kg/m\u00b3)'].mean(),1),  
        round(sum(df_bcm1['Actual cropped area (km\u00b2)']),1),  
        # round(df_bcm1['Cropping intensity'].mean(),1)  
    ]

    df_bcm1.loc[len(df_bcm1.index)] = overall_row

    dfbcm2 = os.path.join(newdir, "Table_summary.csv")
    df_bcm1.to_csv(dfbcm2, index = False)
    print('Saving Average summary table')


     ## FIGURE  - Equity plot ###
    equityplt = os.path.join(newdir, "equity.png")
    fig = plt.figure(figsize = (12,8))
    for i, data in enumerate(summary_csv_data['Equity']):
          plt.plot(years, data, marker='o', label=filtered_LC_crop[i])

    plt.xlabel('Years')
    plt.xticks(years)
    plt.ylabel('Equity (%)')
    plt.title('Equity of Different Crop Classes Over the Years ', fontsize=12)
    plt.legend()
    plt.savefig(equityplt, bbox_inches='tight',pad_inches = 0.1, dpi=100)


    

    r.mask(flags="r")
    r.mask(vector=vectname)
    ### Preparing Adequacy map #########
    maps_adeq=grass.list_grouped(type=['raster'], pattern="adeq_*")['job{}'.format(jobid)]
    r.patch(input=maps_adeq, output='adequacy')
    
    ## FIGURE 3 - adequacy plot ###
    adeqplt = os.path.join(newdir, "adeq.png")
    adeqtif = os.path.join(newdir, "adeq.tif")
    df = gdf.read_file(out1)
    r.out_gdal(input='adequacy', output=adeqtif)
    #print(etmaps)
    adeq=raster2numpy('adequacy', mapset='job{}'.format(jobid))
    adeq = np.ma.masked_where(adeq == -2147483648, adeq)
    fig, ax = plt.subplots(figsize = (12,8))
    plt.imshow(adeq, cmap='jet_r', vmin=0, vmax=np.nanmax(adeq),extent=spatial_extent, interpolation='none', resample=False)
    scalebar = ScaleBar(100, 'km', box_color='w', box_alpha=0.7, location='lower left') # 1 pixel = 0.2 meter
    fig.gca().add_artist(scalebar)
    df.boundary.plot(ax=ax, facecolor='none', edgecolor='k')

    x, y, arrow_length = 1.1, 0.1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
    #ax.legend(bbox_to_anchor=(0.17,0.2))
    plt.colorbar(shrink=0.50, label='Percentage')
    plt.xlabel('Longitude ($^{\circ}$ East)', fontsize=12)  # add axes label
    plt.ylabel('Latitude ($^{\circ}$ North)', fontsize=12)
    plt.title('Adequacy ', fontsize=12)
    plt.savefig(adeqplt, bbox_inches='tight',pad_inches = 0, dpi=100)


    progress += 10
    self.update_state(state='PROGRESS', meta={'current': progress, 'total': total_steps})






    
    

# df_comparison = pd.DataFrame({'Year': years, 'chirps': chirps, 'gpm': gpm, 'gsmap': gsmap, 'era5': era5, 'imd':imd, 'ensind': ensind}, index=years)
    df_pcp = pd.DataFrame({'Year': years, 'PCP_a':imd,'PCP_s':imd_s}, index=years)

    df_eta = pd.DataFrame({'Year': years, 'ETa_a': eta, 'ETa_s': eta_s}, index=years)
    
    # ### Bar chart comparison ETa
    # etabar = os.path.join(newdir, "etabar.png")
    # fig, ax = plt.subplots()
    # if df_comparison['wapor2'].isnull().all() == True or df_comparison['enset'].isnull().all() == True:
        # df_comparison.plot.bar(y = ['ssebop', 'wapor3', 'ensetglobal' ], rot = 40, ax = ax, color=['seagreen', 'limegreen', 'springgreen'])
    # else:
        # df_comparison.plot.bar(y = ['ssebop', 'wapor2', 'wapor3', 'enset', 'ensetglobal'], rot = 40, ax = ax, color=['seagreen', 'limegreen', 'springgreen', 'green', 'lightgreen']) 
    # #ax.invert_yaxis()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.set_title('Annual ETa')
    # ax.set_ylabel('mm/year')
    # plt.savefig(etabar, bbox_inches='tight',pad_inches = 0.1, dpi=100)



    ### Bar chart annual PCP
    pcpbar1 = os.path.join(newdir, "pcpbar1.png")
    fig, ax = plt.subplots()
#     df_pcp.plot.bar(y = ['PCP_a','PCP_s'], rot = 40, ax = ax, color=['seagreen', 'dodgerblue'])
    df_pcp.plot.bar(y = ['PCP_a','PCP_s'],rot = 40, ax = ax, color=['seagreen', 'dodgerblue'])
    #ax.invert_yaxis()
    ax.set_title('Annual and Seasonal Precipitation')
    ax.set_ylabel('mm')
    plt.savefig(pcpbar1, bbox_inches='tight',pad_inches = 0.1, dpi=100)
    
    ### Bar chart annual ETa
    etabar = os.path.join(newdir, "etabar.png")
    fig, ax = plt.subplots()
    df_eta.plot.bar(y = ['ETa_a','ETa_s'], rot = 40, ax = ax, color=['seagreen', 'dodgerblue'])
    #ax.invert_yaxis()
    ax.set_title('Annual and Seasonal EvapoTranspiration')
    ax.set_ylabel('mm')
    plt.savefig(etabar, bbox_inches='tight',pad_inches = 0.1, dpi=100)
    
    ### Bar chart comparison PCP
#     pcpbar2 = os.path.join(newdir, "pcpbar2.png")
#     fig, ax = plt.subplots()
#     df_comparison.plot.bar(y = ['chirps', 'gpm', 'gsmap', 'era5','imd', 'ensind'], rot = 40, ax = ax, color=['dodgerblue', 'dimgrey', 'mediumorchid', 'teal', 'navy', 'azure'])
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     #ax.invert_yaxis()
#     ax.set_title('Comparison of P products')
#     ax.set_ylabel('mm/year')
#     plt.savefig(pcpbar2, bbox_inches='tight',pad_inches = 0.1, dpi=100)
#     Pcomptable = os.path.join(newdir, "Pcomparison.csv")
#     df_comparison.to_csv(Pcomptable, index = True)
#     print('Saving  P comparison table')

    # Table 1 - Saving the annual PCP and ETa into table:
    ## below eta and pcp in km3 vol

    annpcp_vol = [round(float(x / 1000 * studyarea), 2) for x in imd_s]
    anneta_vol = [round(float(x / 1000 * studyarea), 2) for x in eta_s]
    annetb_vol = [round(float(x / 1000 * studyarea), 2) for x in etb_s]
#     annpeta = [a - b for a, b in zip(eval(precip), eta)]
#     annpeta_vol = [round(float(x / 1000000 * studyarea), 2) for x in annpeta]
#     annetr_vol = [round(float(x / 1000000 * studyarea), 2) for x in ssebopetr]

#     df_yearly = pd.DataFrame({'P(mm/year)': eval(precip), 'ETa(mm/year)': eta, 'P-ETa(mm/year)': annpeta, 'ETr(mm/year)': ssebopetr}, index=years)
    df_yearly = pd.DataFrame({'P(mm)': imd_s, 'ETa(mm)': eta_s, 'ETb(mm)':etb_s,'ETr(mm)': ssebopetr_s}, index=years)
    df_yearly.loc['Average'] = round(df_yearly.mean(), 1)
    df_yearly['P(mm)'] = df_yearly['P(mm)'].apply(np.int64)
    df_yearly['ETa(mm)'] = df_yearly['ETa(mm)'].apply(np.int64)
#     df_yearly['P-ETa(mm)'] = df_yearly['P-ETa(mm)'].apply(np.int64)
    df_yearly['ETr(mm)'] = df_yearly['ETr(mm)'].apply(np.int64)
    dfyearly = os.path.join(newdir, "Table1.csv")
    df_yearly.to_csv(dfyearly, index = True)


#     df_yearly2 = pd.DataFrame({ 'P(km\u00b3/year)': annpcp_vol,  'ETa(km\u00b3/year)': anneta_vol, 'P-ETa(km\u00b3/year)': annpeta_vol}, index=years)
    df_yearly2 = pd.DataFrame({ 'P(1000 m\u00b3)': annpcp_vol,  'ETa(1000 m\u00b3)': anneta_vol,'ETb(1000 m\u00b3)': annetb_vol}, index=years)
    df_yearly2.loc['Average'] = round(df_yearly2.mean(), 1)
    dfyearly = os.path.join(newdir, "Table2.csv")
    df_yearly2.to_csv(dfyearly, index = True)

    # # Table 4 for Ea and Ta
    # annea_vol = [round(float(x / 1000000 * studyarea), 2) for x in ea]
    # annta_vol = [round(float(x / 1000000 * studyarea), 2) for x in ta]
    # df_yearly1 = pd.DataFrame({'Ea(mm/year)': ea, 'Ta(mm/year)': ta, 'Ea(km\u00b3/year)': annea_vol,  'Ta(km\u00b3/year)': annta_vol}, index=years)
    # df_yearly1.loc['Average'] = round(df_yearly1.mean(), 1)
    # df_yearly1['Ea(mm/year)'] = df_yearly1['Ea(mm/year)'].apply(np.int64)
    # df_yearly1['Ta(mm/year)'] = df_yearly1['Ta(mm/year)'].apply(np.int64)
    # dfyearly1 = os.path.join(newdir, "Table4.csv")
    # df_yearly1.to_csv(dfyearly1, index = True)

    # PKA trend plots:
    anneta_vol1 = [round(float(x / 1000000 * studyarea), 4) for x in eta]
    print(anneta_vol1)
    annpcp_vol1 = [round(float(x / 1000000 * studyarea), 4) for x in eval(precip)]
    print(annpcp_vol1)
    etavol_avg = statistics.mean(anneta_vol1)
    print(etavol_avg)
    pcpvol_avg = statistics.mean(annpcp_vol1)
    print(pcpvol_avg)
    eta_anomaly = [int(((x / etavol_avg) - 1) * 100) for x in anneta_vol1]
    pcp_anomaly = [int(((x / pcpvol_avg) - 1) * 100) for x in annpcp_vol1]
    df_anomaly = pd.DataFrame({'Year': years, 'ETa_anomaly': eta_anomaly, 'PCP_anomaly': pcp_anomaly}, index=years)
    panoplot = os.path.join(newdir, "panoplot.png")
    etanoplot = os.path.join(newdir, "etanoplot.png")
    fig, ax = plt.subplots()
    x = df_anomaly['Year']
    y = df_anomaly['ETa_anomaly']
    plt.scatter(x, y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    ax.set_ylabel('Delta ET')
    ax.set_xlabel('Year')
    plt.savefig(etanoplot, bbox_inches='tight',pad_inches = 0.1, dpi=100)
    fig, ax = plt.subplots()
    y = df_anomaly['PCP_anomaly']
    plt.scatter(x, y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    ax.set_ylabel('Delta P')
    ax.set_xlabel('Year')
    plt.savefig(panoplot, bbox_inches='tight',pad_inches = 0.1, dpi=100)

    ### Bar chart annual PCP-ET
#     pminetbar = os.path.join(newdir, "pminetbar.png")
#     fig, ax = plt.subplots()
#     df_yearly['positive'] = df_yearly['P-ETa(mm/year)'] > 0
#     df_yearly['P-ETa(mm/year)'].plot(kind='bar', rot = 40, ax = ax, color=df_yearly.positive.map({True: 'deepskyblue', False: 'darkred'}))
#     #ax.invert_yaxis()
#     ax.set_title('Annual P-ETa')
#     ax.set_ylabel('mm/year')
#     plt.savefig(pminetbar, bbox_inches='tight',pad_inches = 0.1, dpi=100)

    # PCP long term trend
    #matplotlib.rcParams.update({'font.size': 26})
    longyears = list(range(1981,2024))
    longyears_str = [str(s) for s in longyears]
    ts_chirps = ["pcpa_chirps_" + s for s in longyears_str]
    chirpsts=[]
    for i in ts_chirps:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = int(round(float(stats['mean'])))
            #mean = float(stats['mean'])
            chirpsts.append(mean)
    print(chirpsts)
    df_pcpts = pd.DataFrame({'Year': longyears, 'PCP': chirpsts}, index=longyears)
    pcptstable = os.path.join(newdir, "pcpts.csv")
    df_pcpts.to_csv(pcptstable, index = False)
    print('Saving  long term precip table')
    pcptsbar = os.path.join(newdir, "pcpbarts.png")
    fig, ax = plt.subplots(figsize=(30,10))
    ax.tick_params(axis='both', labelsize=14)
    #ax.invert_yaxis()
    ax.set_title('Annual Precipitation from 1981 to 2023', fontsize=20)
    ax.set_xlabel('mm/year', fontsize=16)
    x = df_pcpts['Year']
    y = df_pcpts['PCP']
    plt.bar(x, y, color=['dodgerblue'])
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    plt.savefig(pcptsbar, bbox_inches='tight', pad_inches = 0.1, dpi=100)

    progress += 10
    self.update_state(state='PROGRESS', meta={'current': progress, 'total': total_steps})

    ## Bar chart monthly ETa/PCP 

    if et == 'nrsc':
        folder = 'nrsc_et'
    else:
        folder = 'data_monthly'

#     maps_monthly_eta = grass.list_grouped(type=['raster'], pattern=f'{et}_eta_2020*')[folder]

    maps_monthly_pcp=[]
    maps_monthly_eta=[]
    for m in range(1, 13):
          maps_monthly_pcp.append(f"pcpm_imd_avg_{str(m).zfill(2)}")
          maps_monthly_eta.append(f"wapor3_eta_avg_{str(m).zfill(2)}")


    monthly_eta1=[]
    for i in maps_monthly_eta:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = round(float(stats['mean']), 0)
            monthly_eta1.append(mean)

#     if et == 'wapor2' or et == 'wapor3':
#         monthly_eta = [x * 0.1 for x in monthly_eta1]
#     else:
#         monthly_eta = monthly_eta1
    print('monthly_eta:')
    print(monthly_eta1)


#     maps_monthly_pcp=grass.list_grouped(type=['raster'], pattern=f'pcpm_{precip}_2020*')['data_monthly']


          
    monthly_pcp=[]
    for i in maps_monthly_pcp:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = round(float(stats['mean']), 0)
            monthly_pcp.append(mean)
    print(monthly_pcp)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_monthly = pd.DataFrame({'ETa': monthly_eta1, 'PCP': monthly_pcp}, index=months)

    monthlyetabar = os.path.join(newdir, "monthlyetabar.png")
    fig, ax = plt.subplots()
    df_monthly.plot.bar(y = ['ETa', 'PCP'], rot = 40, ax = ax, color=['seagreen', 'dodgerblue'])
    #ax.invert_yaxis()
    ax.set_title('Monthly variation of ETa and P')
    ax.set_ylabel('mm/month')
    plt.savefig(monthlyetabar, bbox_inches='tight',pad_inches = 0.1, dpi=100)

    

    
    # r.mask(raster="LC_studyarea", maskcats='2 thru 7')

    r.mask(vector=vectname)
    maps_grace=grass.list_grouped(type=['raster'], pattern="mascon_lwe_thickness*")['grace']
    grace=[]
    grace_dt=[]
    grace_yr=[]
    for i in maps_grace:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = float(stats['mean'])
            yr = i.split('_')[3]
            mm = i.split('_')[4]
            dt = yr + '-' + mm
            grace.append(mean)
            grace_yr.append(yr)
            grace_dt.append(dt)


    dfgrace = pd.DataFrame({'waterlevel': grace, 'date': grace_dt, 'year': grace_yr}, index=grace_yr)

    if studyarea >= 10000:
        grace_fig = os.path.join(newdir, "grace_fig.png")
        fig, ax = plt.subplots()
        dfgrace.plot.line(y = ['waterlevel', 'year'], rot = 40, ax = ax, color=['black'])
        ax.set_title('Change in water storage')
        ax.set_ylabel('Equivalent cm of water')
        plt.savefig(grace_fig, bbox_inches='tight',pad_inches = 0.1, dpi=100)
        gracetable = os.path.join(newdir, "gracetable.csv")
        dfgrace.to_csv(gracetable, index = False)
        print('Saving  grace table')
    else:
        print('No grace data - area is small')
    
    r.mask(flags="r")
    
    g.region(vector=vectname, res=0.003)
    r.mask(vector=vectname)
    ### Climate change analysis
    ### SSP245
    yearcc1 = list(range(2015,2061))
    yearcc = [str(s) for s in yearcc1]
    maps_tdegssp245 = ["tdegDev_annual_ssp245_" + s for s in yearcc]
    tdegssp245=[]
    for i in maps_tdegssp245:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = round(float(stats['mean']), 1)
            tdegssp245.append(mean)
    print('tdegssp245:')
    print(tdegssp245)

    maps_prssp245 = ["prDev_annual_ssp245_" + s for s in yearcc]
    prssp245=[]
    for i in maps_prssp245:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = int(round(float(stats['mean'])))
            prssp245.append(mean)
    print('prssp245:')
    print(prssp245)

    ### SSP585
    maps_tdegssp585 = ["tdegDev_annual_ssp585_" + s for s in yearcc]
    tdegssp585=[]
    for i in maps_tdegssp585:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = round(float(stats['mean']), 1)
            tdegssp585.append(mean)
    print('tdegssp585:')
    print(tdegssp585)

    maps_prssp585 = ["prDev_annual_ssp585_" + s for s in yearcc]
    prssp585=[]
    for i in maps_prssp585:
            stats = grass.parse_command('r.univar', map=i, flags='g')
            mean = int(round(float(stats['mean'])))
            prssp585.append(mean)
    print('prssp585:')
    print(prssp585)
   
    df_cc1 = pd.DataFrame({'Year': yearcc, 'tdegssp245': tdegssp245, 'prssp245': prssp245, 'tdegssp585': tdegssp585, 'prssp585': prssp585}, index=yearcc)

    df_cc1['tdegssp245_mn'] = df_cc1['tdegssp245'].rolling(5).mean()
    df_cc1['tdegssp245_std'] = df_cc1['tdegssp245'].rolling(5).std()
    df_cc1['tdegssp245_un'] = df_cc1['tdegssp245_mn'] - df_cc1['tdegssp245_std']
    df_cc1['tdegssp245_ov'] = df_cc1['tdegssp245_mn'] + df_cc1['tdegssp245_std']

    df_cc1['prssp245_mn'] = df_cc1['prssp245'].rolling(5).mean()
    df_cc1['prssp245_std'] = df_cc1['prssp245'].rolling(5).std()
    df_cc1['prssp245_un'] = df_cc1['prssp245_mn'] - df_cc1['prssp245_std']
    df_cc1['prssp245_ov'] = df_cc1['prssp245_mn'] + df_cc1['prssp245_std']

    df_cc1['tdegssp585_mn'] = df_cc1['tdegssp585'].rolling(5).mean()
    df_cc1['tdegssp585_std'] = df_cc1['tdegssp585'].rolling(5).std()
    df_cc1['tdegssp585_un'] = df_cc1['tdegssp585_mn'] - df_cc1['tdegssp585_std']
    df_cc1['tdegssp585_ov'] = df_cc1['tdegssp585_mn'] + df_cc1['tdegssp585_std']

    df_cc1['prssp585_mn'] = df_cc1['prssp585'].rolling(5).mean()
    df_cc1['prssp585_std'] = df_cc1['prssp585'].rolling(5).std()
    df_cc1['prssp585_un'] = df_cc1['prssp585_mn'] - df_cc1['prssp585_std']
    df_cc1['prssp585_ov'] = df_cc1['prssp585_mn'] + df_cc1['prssp585_std']

    df_cc = df_cc1.dropna()
    mk1 = mk.original_test(df_cc['tdegssp245'])
    mk11 = pd.DataFrame(mk1, columns=['Name'])
    t1 = mk11.loc[0,'Name']
    sl1 = round(float(mk11.loc[7,'Name']),2)
    in1 = round(float(mk11.loc[8,'Name']),2)
    mk2 = mk.original_test(df_cc['prssp245'])
    mk21 = pd.DataFrame(mk2, columns=['Name'])
    t2 = mk21.loc[0,'Name']
    sl2 = round(float(mk21.loc[7,'Name']),2)
    in2 = round(float(mk21.loc[8,'Name']),2)
    mk3 = mk.original_test(df_cc['tdegssp585'])
    mk31 = pd.DataFrame(mk3, columns=['Name'])
    t3 = mk31.loc[0,'Name']
    sl3 = round(float(mk31.loc[7,'Name']),2)
    in3 = round(float(mk31.loc[8,'Name']),2)
    mk4 = mk.original_test(df_cc['prssp585'])
    mk41 = pd.DataFrame(mk4, columns=['Name'])
    t4 = mk41.loc[0,'Name']
    sl4 = round(float(mk41.loc[7,'Name']),2)
    in4 = round(float(mk41.loc[8,'Name']),2)


    figt1 = os.path.join(newdir, "figt1.png")
    fig = plt.figure()
    x = df_cc['Year'].astype(int)
    y = df_cc['tdegssp245_mn']
    plt.plot(x,y, color='green')
    plt.ylabel('Temperature (DegC)')
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
#     plt.fill_between(x, df_cc['tdegssp245_un'], df_cc['tdegssp245_ov'], color='b', alpha=.1)
    plt.title('Annual mean Temperature (deviation)') 
    plt.savefig(figt1, bbox_inches='tight',pad_inches = 0.1, dpi=100)

    figt2 = os.path.join(newdir, "figt2.png")
    fig = plt.figure()
    x = df_cc['Year'].astype(int)
    y = df_cc['prssp245_mn']
    plt.plot(x,y)
    plt.ylabel('Precipitation (mm/year)')
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
#     plt.fill_between(x, df_cc['prssp245_un'], df_cc['prssp245_ov'], color='b', alpha=.1)
    plt.title('Annual mean Precipitation (deviation)')
    plt.savefig(figt2, bbox_inches='tight',pad_inches = 0.1, dpi=100)

    figt3 = os.path.join(newdir, "figt3.png")
    fig = plt.figure()
    x = df_cc['Year'].astype(int)
    y = df_cc['tdegssp585_mn']
    plt.plot(x,y, color='green')
    plt.ylabel('Temperature (DegC)')
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
#     plt.fill_between(x, df_cc['tdegssp585_un'], df_cc['tdegssp585_ov'], color='b', alpha=.1)
    plt.title('Annual mean Temperature (deviation)') 
    plt.savefig(figt3, bbox_inches='tight',pad_inches = 0.1, dpi=100)

    figt4 = os.path.join(newdir, "figt4.png")
    fig = plt.figure()
    x = df_cc['Year'].astype(int)
    y = df_cc['prssp585_mn']
    plt.plot(x,y)
    plt.ylabel('Precipitation (mm/year)')
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
#     plt.fill_between(x, df_cc['prssp585_un'], df_cc['prssp585_ov'], color='b', alpha=.1)
    plt.title('Annual mean Precipitation (deviation)')
    plt.savefig(figt4, bbox_inches='tight',pad_inches = 0.1, dpi=100)

    r.mask(flags="r")
    plt.close('all')

    progress += 10
    self.update_state(state='PROGRESS', meta={'current': progress, 'total': total_steps})

    ### Parse statistics to the report
    stats = dict(et=et,pcp=precip,st_yr=start_yr,end_yr=end_yr,st_mo=start_month_name,end_mo=end_month_name,state=state, centx=centX,centy=centY,w=west,e=east,n=north,s=south,area=studyarea,dem_min=dem_min,dem_max=dem_max,lc0=LC0,lc1=LC1,lca0=LCA0,lca1=LCA1,eta=mean_eta_basin,p=mean_pcp_basin,etr=mean_etr_basin,t1=t1,t2=t2,t3=t3,t4=t4,sl1=sl1,sl2=sl2,sl3=sl3,sl4=sl4,
                 plots_width=plots_width,plots_height=plots_height,
                 lcc_year=selected_lcc_year,
                 webImg_style=webImg_style,
                 pdfImg_style=pdfImg_style,
                 pdfImg_style2=pdfImg_style2,
                 )
    #etb=mean_etb_basin,etg=mean_etg_basin
    mean = 100

    ## Mimic an empty mapset with WIND file for raster2numpy to work.
    mapdir = os.path.join(settings.GRASS_DB, 'wagen', 'job{}'.format(jobid))
    windsrc = os.path.join(mapdir, 'WIND')
    winddst = os.path.join(newdir, 'WIND')
    shutil.copy2(windsrc, winddst)
    user.close()
    os.mkdir(mapdir)
    shutil.copy2(winddst, windsrc)

    htmlfile1 = render_prod_html(jobid, myarea, stats)

    htmlfile2 = render_pdf_html(jobid, myarea, stats)
    pdffile = render_pdf(htmlfile2, jobid)
    print("Preparing report !")

    progress += 10
    self.update_state(state='PROGRESS', meta={'current': progress, 'total': total_steps})

#     base_url = settings.BASE_URL

#     sub="Water Accounting Report"
#     mess = f"Your requested Water Accounting report is ready. You can access the report using this link: {base_url}/media/{jobid}/index.html"
    

#     to=current_user

#     attach=pdffile
    #send_mail_attach(sub, mess, to, attach)
    return htmlfile1, pdffile


